"""Reconstruct a video from its first k tokens per frame and write a side-by-side mp4.

    python scripts/reconstruct.py --ckpt-root /path/to/checkpoints --tokenizer k600-semantok \
        --video clip.mp4 --ks 1 4 16 64 256 --out recon.mp4

The clip is read with the Kinetics recipe (centre 4 s, 17 frames, centre square, 128 px); pass
``--uco3d`` for the uCO3D recipe. Columns: VAE reconstruction, then one column per k.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from semantok import Tokenizer, hub  # noqa: E402
from semantok.data.video import load_kinetics_clip, load_uco3d_clip  # noqa: E402
from semantok.video_io import write_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--tokenizer", default="k600-semantok")
    ap.add_argument("--video", required=True)
    ap.add_argument("--uco3d", action="store_true")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reconstruction.mp4")
    args = ap.parse_args()

    tok = Tokenizer(hub.join(args.ckpt_root, "tokenizers", args.tokenizer))
    clip = (load_uco3d_clip if args.uco3d else load_kinetics_clip)(args.video).unsqueeze(0)
    lat = tok.vae_encode(clip)
    tokens = tok.encode(clip, latents=lat)
    cols = [tok.vae_decode(lat)[0]] + [tok.decode(tokens, k=k, seed=args.seed)[0] for k in args.ks]
    write_grid(args.out, [cols])
    print(f"wrote {args.out}: columns = VAE, " + ", ".join(f"k={k}" for k in args.ks))


if __name__ == "__main__":
    with torch.no_grad():
        main()
