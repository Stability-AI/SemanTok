"""AR generation evaluation (paper Fig. 4): FVD, FID, ViCLIP, ClipV (+ class accuracy on Kinetics).

    # Kinetics-600, class-to-video, SemanTok tokenizer + 201M AR, budgets k = 1 ... 256
    python scripts/eval_generation.py --ckpt-root /path/to/checkpoints --ar k600-semantok-d16 \
        --k600-root /data/kinetics600/val --ks 1 4 8 16 32 64 128 256 --out results/

    # uCO3D, text-to-video (ID and OOD pools, pooled at the official 1014:152 ratio)
    python scripts/eval_generation.py --ckpt-root ... --ar uco3d-semantok-d16 \
        --uco3d-root /data/uco3d --uco3d-index /data/uco3d/index.csv --ks 16 --out results/

``--ckpt-root`` is a local directory or ``hf://<org>/<repo>`` laid out as ``tokenizers/<name>/``
and ``ar/<name>/``. Sample counts, guidance and seeds default to the paper's protocol.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from semantok import Tokenizer, hub, load_ar  # noqa: E402
from semantok.eval import runner  # noqa: E402
from semantok.eval.pools import K600Pool, UCO3DPool  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--ar", required=True, help="AR checkpoint name, e.g. k600-semantok-d16")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--k600-root", help="Kinetics-600 validation videos: <root>/<label>/<file>.mp4")
    ap.add_argument("--uco3d-root", help="uCO3D root; videos at <root>/<video_path>")
    ap.add_argument("--uco3d-index", help="uCO3D index CSV (category, sequence_name, video_path, captions)")
    ap.add_argument("--n-samples", type=int, help="override the paper's sample count per k")
    ap.add_argument("--cfg", type=float, help="override the paper's AR guidance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-models", help="third-party metric models (scripts/download_eval_models.sh)")
    ap.add_argument("--cache-dir", default="cache", help="reference features are cached here")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda"
    ar, ar_cfg = load_ar(hub.join(args.ckpt_root, "ar", args.ar), device)
    tok = Tokenizer(hub.join(args.ckpt_root, "tokenizers", ar_cfg["tokenizer"]), device)
    k600 = ar_cfg["cond"] == "class"
    ex = runner.Extractors(device, args.eval_models, with_umt=k600)
    os.makedirs(args.out, exist_ok=True)

    if k600:
        pools = {"val": K600Pool(args.k600_root)}
        text_encoder = None
    else:
        from semantok.text import UMT5TextEncoder
        pools = {s: UCO3DPool(args.uco3d_root, args.uco3d_index, f"uco3d_{s}_1024") for s in ("id", "ood")}
        text_encoder = UMT5TextEncoder(device)
    refs = {s: runner.reference_features(tok, p, ex, args.cache_dir, args.workers) for s, p in pools.items()}

    results = {"ar": args.ar, "tokenizer": ar_cfg["tokenizer"], "seed": args.seed, "per_k": {}}
    for k in args.ks:
        if k600:
            n = args.n_samples or runner.K600_N_SAMPLES.get(k, 2048)
            cfg = args.cfg if args.cfg is not None else runner.K600_AR_CFG[k]
        else:
            n = args.n_samples or runner.UCO3D_N_SAMPLES
            cfg = args.cfg if args.cfg is not None else runner.UCO3D_AR_CFG
        split_scores, gens = {}, {}
        for s, pool in pools.items():
            split_scores[s], gens[s] = runner.eval_generation(
                ar, tok, pool, ex, k, n, cfg, refs[s], seed=args.seed, text_encoder=text_encoder)
            print(f"[{args.ar}] {s} k={k}: {split_scores[s]}", flush=True)
        entry = split_scores["val"] if k600 else {"pooled": runner.pool_uco3d(split_scores, refs, gens),
                                                   **split_scores}
        results["per_k"][str(k)] = entry
        with open(os.path.join(args.out, f"generation_{args.ar}.json"), "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results["per_k"], indent=2))


if __name__ == "__main__":
    main()
