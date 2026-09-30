"""Generate videos with an AR model at several token budgets and write an mp4 grid.

    # class-to-video (Kinetics-600): rows = samples, columns = k
    python scripts/generate.py --ckpt-root /path/to/checkpoints --ar k600-semantok-d16 \
        --class-name "yoga" --n 4 --ks 4 16 64 256 --out yoga.mp4

    # text-to-video (uCO3D)
    python scripts/generate.py --ckpt-root ... --ar uco3d-semantok-d16 \
        --prompt "A small orange basketball on a plaid tablecloth" --n 4 --ks 4 16 64 --out ball.mp4

Guidance defaults to the paper's value for each dataset and k (``--cfg`` overrides it).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from semantok import Tokenizer, hub, load_ar  # noqa: E402
from semantok.eval.runner import K600_AR_CFG, UCO3D_AR_CFG  # noqa: E402
from semantok.tokens import T_LAT, unflatten_time_first  # noqa: E402
from semantok.video_io import write_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--ar", default="k600-semantok-d16")
    ap.add_argument("--class-name", help="Kinetics-600 class (class-conditioned models)")
    ap.add_argument("--prompt", help="caption (text-conditioned models)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 16, 64, 256])
    ap.add_argument("--cfg", type=float)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="generation.mp4")
    args = ap.parse_args()

    device = "cuda"
    ar, cfg = load_ar(hub.join(args.ckpt_root, "ar", args.ar), device)
    tok = Tokenizer(hub.join(args.ckpt_root, "tokenizers", cfg["tokenizer"]), device)
    kw = {}
    if cfg["cond"] == "class":
        if args.class_name not in cfg["classes"]:
            raise SystemExit(f"--class-name must be one of the {len(cfg['classes'])} classes in the checkpoint config")
        kw["class_id"] = torch.full((args.n,), cfg["classes"].index(args.class_name), device=device)
    else:
        from semantok.text import UMT5TextEncoder
        if not args.prompt:
            raise SystemExit("--prompt is required for a text-conditioned model")
        emb, mask = UMT5TextEncoder(device).embed([args.prompt] * args.n, cfg.get("max_text_len", 128))
        kw.update(text_emb=emb, text_mask=mask)

    rows = [[] for _ in range(args.n)]
    for k in args.ks:
        guidance = args.cfg if args.cfg is not None else (
            K600_AR_CFG.get(k, 1.0) if cfg["cond"] == "class" else UCO3D_AR_CFG)
        g = torch.Generator(device=device).manual_seed(args.seed + 1000 * k)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = ar.sample(args.n, length=T_LAT * k, cfg=guidance, temperature=args.temperature,
                            device=device, generator=g, **kw)
        videos = tok.decode(unflatten_time_first(ids), k=k, seed=args.seed)
        for i, v in enumerate(videos):
            rows[i].append(v)
    write_grid(args.out, rows)
    print(f"wrote {args.out}: rows = samples, columns = " + ", ".join(f"k={k}" for k in args.ks))


if __name__ == "__main__":
    with torch.no_grad():
        main()
