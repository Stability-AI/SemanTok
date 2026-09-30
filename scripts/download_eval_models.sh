#!/usr/bin/env bash
# Fetch the third-party evaluation models into $SEMANTOK_EVAL_MODELS (default ~/.cache/semantok/eval_models).
#
#   i3d_torchscript.pt                        FVD (StyleGAN-V export of the TF-Hub I3D)
#   ViCLIP-L_InternVid-FLT-10M.pth            ViCLIP (HF OpenGVLab/ViCLIP, gated: accept the terms and `huggingface-cli login`)
#   l16_ptk710_ftk710_ftk600_f16_res224.pth   UMT-L K600 classifier (HF OpenGVLab/UMT, gated)
#   InternVideo/Data/InternVid/viclip         ViCLIP model code (InternVideo, Apache-2.0)
#   VBench/vbench/third_party/umt             UMT model code (VBench, Apache-2.0)
#
# FID uses torchvision's InceptionV3 weights, which torchvision downloads by itself.
set -euo pipefail
DIR="${SEMANTOK_EVAL_MODELS:-$HOME/.cache/semantok/eval_models}"
mkdir -p "$DIR" && cd "$DIR"

INTERNVIDEO_REV=3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3
VBENCH_REV=fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490

sha_ok() { [[ -f "$1" ]] && [[ "$(sha256sum "$1" | cut -d' ' -f1)" == "$2" ]]; }

if ! sha_ok i3d_torchscript.pt bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2; then
  curl -L -o i3d_torchscript.pt "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
  sha_ok i3d_torchscript.pt bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2 \
    || { echo "i3d_torchscript.pt checksum mismatch" >&2; exit 1; }
fi

if ! sha_ok ViCLIP-L_InternVid-FLT-10M.pth d49b83b8c236e65778aaed3824053e1d930ea1993706d171b574a8bdb69ceecf; then
  huggingface-cli download OpenGVLab/ViCLIP ViCLIP-L_InternVid-FLT-10M.pth --local-dir .
fi

if ! sha_ok l16_ptk710_ftk710_ftk600_f16_res224.pth 79e7c8ab429d2c1c5d783dd90915d4bd11235937598b8aafd39d672ce04e112b; then
  huggingface-cli download OpenGVLab/UMT single_modality/l16_ptk710_ftk710_ftk600_f16_res224.pth --local-dir .
  mv single_modality/l16_ptk710_ftk710_ftk600_f16_res224.pth . && rmdir single_modality
fi

sparse_clone() {  # url rev dir subpath
  if [[ ! -d "$3" ]]; then
    git clone --filter=blob:none --no-checkout "$1" "$3"
    git -C "$3" sparse-checkout set "$4"
    git -C "$3" checkout -q "$2"
  fi
}
sparse_clone https://github.com/OpenGVLab/InternVideo.git "$INTERNVIDEO_REV" InternVideo Data/InternVid
sparse_clone https://github.com/Vchitect/VBench.git "$VBENCH_REV" VBench vbench/third_party/umt

echo "eval models ready in $DIR"
