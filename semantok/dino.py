"""Frozen DINOv2-L features for the SemanTok encoder input.

SemanTok's encoder reads, per latent frame, a DINOv2-L patch grid ``[16, 16, 1024]`` concatenated
channel-wise with the 16-channel VidTok latent, and adds a projection of a DINOv2 CLS vector to the
first register (the k=1 token) of each frame.

17 RGB frames map to 5 causal latent frames as ``{0}, {1..4}, {5..8}, {9..12}, {13..16}``. A latent
frame's grid is either the DINO grid of the first frame of its group (``agg="first"``) or the mean
over the group (``agg="avg"``).
"""
import torch
import torch.nn.functional as F

DINOV2_MODEL = "facebook/dinov2-large"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def causal_frame_groups(num_frames=17, num_latent=5):
    groups = [[0]] + [list(range(4 * (i - 1) + 1, 4 * i + 1)) for i in range(1, num_latent)]
    assert groups[-1][-1] == num_frames - 1, f"{num_frames} frames do not map onto {num_latent} latents"
    return groups


class DinoV2:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        from transformers import Dinov2Model

        self.model = Dinov2Model.from_pretrained(DINOV2_MODEL).to(device).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _hidden(self, frames_m1):
        """``[N, 3, H, W]`` in [-1, 1] -> last hidden state ``[N, 257, 1024]`` (CLS first)."""
        x = (frames_m1.to(self.device).float() + 1.0) / 2.0
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        x = ((x - self.mean) / self.std).to(self.dtype)
        with torch.autocast("cuda", dtype=self.dtype, enabled=str(self.device) != "cpu"):
            return self.model(pixel_values=x).last_hidden_state.float()

    @torch.no_grad()
    def encoder_input(self, clip_m1, agg="first", num_latent=5):
        """One clip ``[3, 17, H, W]`` -> per-latent patch grids ``[5, 16, 16, 1024]``."""
        groups = causal_frame_groups(clip_m1.shape[1], num_latent)
        frames = [g[0] for g in groups] if agg == "first" else list(range(clip_m1.shape[1]))
        h = self._hidden(clip_m1[:, frames].permute(1, 0, 2, 3))[:, 1:]            # [N, 256, 1024]
        g = h.reshape(h.shape[0], 16, 16, h.shape[-1])
        if agg == "first":
            return g
        if agg != "avg":
            raise ValueError(f"agg must be 'first' or 'avg', got {agg!r}")
        return torch.stack([g[grp].mean(0) for grp in groups])

    @torch.no_grad()
    def cls(self, clip_m1, mode="per_frame", num_latent=5):
        """Global vector for register 0: ``[5, 1024]`` (``per_frame``: CLS of the first frame of each
        latent group) or ``[1024]`` (``frame0``: CLS of frame 0, broadcast to every frame)."""
        if mode == "frame0":
            return self._hidden(clip_m1[:, :1].permute(1, 0, 2, 3))[0, 0]
        if mode != "per_frame":
            raise ValueError(f"mode must be 'per_frame' or 'frame0', got {mode!r}")
        idx = [g[0] for g in causal_frame_groups(clip_m1.shape[1], num_latent)]
        return self._hidden(clip_m1[:, idx].permute(1, 0, 2, 3))[:, 0]
