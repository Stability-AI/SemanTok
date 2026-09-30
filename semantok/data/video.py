"""Read a source video into the 17-frame, 128x128 clip the tokenizers were trained and scored on.

Both loaders return ``[3, 17, 128, 128]`` float in ``[-1, 1]``. The temporal window and the
resize are dataset conventions and they change the numbers, so each dataset keeps its own.

Kinetics-600: the centre 4 s of the video (17 samples, first and last 4.0 s apart; shorter videos
span the whole clip), a centre square crop of the native frame, then bicubic (antialiased) resize.

uCO3D: 17 frames spread evenly over [10%, 25%] of the turntable video, torchvision
``Resize(128, bicubic)`` on the short side followed by ``CenterCrop(128)``.
"""
import numpy as np
import torch
import torch.nn.functional as F

NUM_FRAMES = 17
SIZE = 128


def _read(path, idx_fn):
    import decord

    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    idx = idx_fn(len(vr), float(vr.get_avg_fps()) or 0.0)
    batch = vr.get_batch(idx)
    arr = batch if isinstance(batch, torch.Tensor) else torch.from_numpy(batch.asnumpy())
    return arr.permute(3, 0, 1, 2).contiguous().float().div_(255.0)     # [3, T, H, W] in [0, 1]


# ------------------------------------------------------------------------------ Kinetics-600 ---
def kinetics_frame_indices(n_frames, fps, window_sec=4.0, num_frames=NUM_FRAMES):
    if n_frames <= 1:
        return np.zeros(num_frames, dtype=np.int32)
    duration = (n_frames / fps) if fps > 0 else 0.0
    last = n_frames - 1
    if duration <= window_sec + 1e-6:
        return np.linspace(0, last, num_frames, dtype=np.int32)
    span = min(window_sec * fps, float(last))
    lo = 0.5 * (last - span)
    idx = np.linspace(lo, lo + span, num_frames)
    return np.clip(np.rint(idx), 0, last).astype(np.int32)


def load_kinetics_clip(path, size=SIZE):
    fr = _read(path, kinetics_frame_indices)
    _, _, h, w = fr.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    x = fr[:, :, top:top + s, left:left + s].permute(1, 0, 2, 3)             # [T, 3, s, s]
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    return x.clamp(0, 1).permute(1, 0, 2, 3).mul(2.0).sub(1.0)


# ------------------------------------------------------------------------------------- uCO3D ---
def uco3d_frame_indices(n_frames, fps=None, t_lo=0.10, t_hi=0.25, num_frames=NUM_FRAMES):
    lo = int(round(t_lo * (n_frames - 1)))
    hi = int(round(t_hi * (n_frames - 1)))
    return np.linspace(lo, hi, num_frames, dtype=np.int32)


def load_uco3d_clip(path, size=SIZE):
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode

    fr = _read(path, uco3d_frame_indices)
    x = T.CenterCrop(size)(T.Resize(size, interpolation=InterpolationMode.BICUBIC)(fr))
    return (x - 0.5) / 0.5
