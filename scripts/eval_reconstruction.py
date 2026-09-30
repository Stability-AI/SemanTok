"""Tokenizer reconstruction evaluation (paper Table 2): each clip is encoded, decoded from its first k
tokens, and scored.

    python scripts/eval_reconstruction.py --ckpt-root /path/to/checkpoints --tokenizer k600-semantok \
        --k600-root /data/kinetics600/val --ks 1 4 8 16 32 64 128 256 --out results/

Kinetics-600: FVD / FID / ViCLIP / ClipV / class accuracy on the 2048-clip reference pool, and
PSNR / SSIM on a separate 2560-clip pool (``--pixel-pool``). uCO3D: the 1014 ID + 152 OOD official
validation clips, scored against the 1024-clip ID / OOD reference pools and pooled at 1014:152.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantok import Tokenizer, hub  # noqa: E402
from semantok.eval import runner  # noqa: E402
from semantok.eval.pools import K600Pool, UCO3DPool  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--tokenizer", required=True, help="e.g. k600-semantok, uco3d-videoflextok")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--k600-root")
    ap.add_argument("--uco3d-root")
    ap.add_argument("--uco3d-index")
    ap.add_argument("--skip-pixel", action="store_true", help="skip the PSNR / SSIM pool")
    ap.add_argument("--skip-set", action="store_true", help="skip the FVD / FID / ViCLIP / ClipV pools")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-models")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda"
    tok = Tokenizer(hub.join(args.ckpt_root, "tokenizers", args.tokenizer), device)
    k600 = tok.meta["dataset"] == "k600"
    ex = runner.Extractors(device, args.eval_models, with_umt=k600)
    os.makedirs(args.out, exist_ok=True)
    res = {"tokenizer": args.tokenizer, "seed": args.seed}
    out_path = os.path.join(args.out, f"reconstruction_{args.tokenizer}.json")

    def dump():
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)

    if k600:
        if not args.skip_set:
            bank = K600Pool(args.k600_root)
            ref = runner.reference_features(tok, bank, ex, args.cache_dir, args.workers)
            res["set"], _ = runner.eval_reconstruction(tok, bank, ex, args.ks, ref, seed=args.seed,
                                                       workers=args.workers)
            dump()
        if not args.skip_pixel:
            pix = K600Pool(args.k600_root, name="k600_val_pixel_2560")
            res["pixel"], _ = runner.eval_reconstruction(tok, pix, ex, args.ks, seed=args.seed,
                                                         pixel_only=True, workers=args.workers)
            dump()
    else:
        refs, scores, gens = {}, {}, {}
        for s, n in (("id", 1014), ("ood", 152)):
            refs[s] = runner.reference_features(
                tok, UCO3DPool(args.uco3d_root, args.uco3d_index, f"uco3d_{s}_1024"), ex,
                args.cache_dir, args.workers)
            val = UCO3DPool(args.uco3d_root, args.uco3d_index, f"uco3d_val_{s}_{n}")
            scores[s], gens[s] = runner.eval_reconstruction(
                tok, val, ex, args.ks, refs[s], seed=args.seed, pixel_only=args.skip_set,
                workers=args.workers)
        res["per_split"] = scores
        if not args.skip_set:
            res["pooled"] = {k: runner.pool_uco3d({s: scores[s][k] for s in scores}, refs,
                                                  {s: gens[s][k] for s in gens}) for k in args.ks}
        dump()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
