"""Video metrics used in the paper: FVD, FID, ViCLIP text-video (clipT) and video-video (ClipV)
similarity, and UMT-L Kinetics-600 top-1 accuracy.

Every extractor takes one clip ``[3, T, H, W]``. Frechet extractors take values in ``[0, 1]``;
ViCLIP and UMT take ``[-1, 1]`` (the tokenizer's output range). The third-party weights and model
code are fetched by ``scripts/download_eval_models.sh`` into one directory (``--eval-models``).

Preprocessing is part of the metric. Resizing, value range and frame choice below are exactly the
ones behind the paper numbers, and a different choice gives a different (not comparable) number.
"""
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_eval_models_dir():
    return os.environ.get("SEMANTOK_EVAL_MODELS",
                          os.path.join(os.path.expanduser("~"), ".cache", "semantok", "eval_models"))


# ------------------------------------------------------------------------------------ Frechet ---
def mu_sigma(x):
    """(mean, unbiased covariance) of a ``[N, d]`` feature matrix, in float64."""
    x = x.to(torch.float64)
    mu = x.mean(0)
    xc = x - mu
    return mu, (xc.t() @ xc) / max(1, x.shape[0] - 1)


def _sqrtm_psd(mat):
    sym = 0.5 * (mat + mat.transpose(-1, -2))
    vals, vecs = torch.linalg.eigh(sym)
    return (vecs * vals.clamp_min(0).sqrt()) @ vecs.transpose(-1, -2)


def frechet_distance(real, fake, eps=1e-6):
    """Frechet distance between two ``[N, d]`` feature sets (float64, eigh square root)."""
    mu1, s1 = mu_sigma(real)
    mu2, s2 = mu_sigma(fake)
    d = mu1 - mu2
    ident = torch.eye(s1.shape[0], dtype=s1.dtype)
    r1 = _sqrtm_psd(s1 + eps * ident)
    covmean = _sqrtm_psd(r1 @ (s2 + eps * ident) @ r1)
    fd = d.dot(d) + torch.trace(s1) + torch.trace(s2) - 2.0 * torch.trace(covmean)
    return float(fd.clamp_min(0))


# --------------------------------------------------------------------------------- extractors ---
class InceptionFrames:
    """FID features: torchvision InceptionV3 (IMAGENET1K_V1) pool features, one row per frame."""

    name = "fid"

    def __init__(self, device):
        from torchvision.models import Inception_V3_Weights, inception_v3

        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        m.fc = nn.Identity()
        self.m = m.eval().requires_grad_(False).to(device)
        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, video01):
        x = video01.to(self.device).permute(1, 0, 2, 3)              # [T, 3, H, W]
        x = F.interpolate(x, size=299, mode="bilinear", align_corners=False)
        return self.m((x - self.mean) / self.std)                    # [T, 2048]


class I3DFVD:
    """FVD features: the TF-Hub I3D of Unterthiner et al. (2018), StyleGAN-V torchscript, 400-d."""

    name = "fvd"
    SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"

    def __init__(self, device, models_dir=None):
        path = os.path.join(models_dir or default_eval_models_dir(), "i3d_torchscript.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} missing; run scripts/download_eval_models.sh")
        self.m = torch.jit.load(path, map_location=device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, video01):
        # The script's `rescale` expects [0, 255]; feeding [0, 1] maps every clip to ~-1 and FVD
        # collapses towards 0 without any error.
        x = (video01.to(self.device).float().unsqueeze(0) * 255.0).contiguous()
        b, c, t, h, w = x.shape
        if (h, w) != (224, 224):
            x = F.interpolate(x.reshape(b, c * t, h, w), size=(224, 224), mode="bilinear",
                              align_corners=False).reshape(b, c, t, 224, 224).contiguous()
        return self.m(x, rescale=True, resize=False, return_features=True).float().reshape(1, -1)


class ViCLIP:
    """ViCLIP-L (InternVid-FLT-10M): 8 of 17 frames at stride 2, 224 px, L2-normalised embeddings."""

    N_FRAMES = 8

    def __init__(self, device, models_dir=None):
        d = models_dir or default_eval_models_dir()
        code_dir = os.path.join(d, "InternVideo", "Data", "InternVid")
        weights = os.path.join(d, "ViCLIP-L_InternVid-FLT-10M.pth")
        for p in (os.path.join(code_dir, "viclip"), weights):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} missing; run scripts/download_eval_models.sh")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        _stub_viclip_imports()
        from viclip import ViCLIP as _ViCLIP
        from viclip.simple_tokenizer import SimpleTokenizer

        self.tokenizer = SimpleTokenizer()
        self.model = _ViCLIP(tokenizer=self.tokenizer, size="l", pretrain="").eval()
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        miss, unexp = self.model.load_state_dict(sd, strict=False)
        if miss or unexp:
            raise RuntimeError(f"ViCLIP state-dict mismatch: missing={miss[:3]} unexpected={unexp[:3]}")
        # fp32 weights under bf16 autocast: the text tower forces LayerNorm inputs to fp32, so
        # casting the weights to half precision raises.
        self.model = self.model.to(device).requires_grad_(False)
        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
        self._text_cache = {}

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device) != "cpu")

    @torch.no_grad()
    def text_features(self, captions):
        """``[N, 768]`` L2-normalised text embeddings (cached by string)."""
        for c in dict.fromkeys(captions):
            if c not in self._text_cache:
                with self._autocast():
                    f = self.model.get_text_features(c, self.tokenizer, {})
                self._text_cache[c] = F.normalize(f.float().reshape(-1), dim=-1).cpu()
        return torch.stack([self._text_cache[c] for c in captions])

    @torch.no_grad()
    def video_features(self, clip_m1):
        """One clip ``[3, T, H, W]`` in [-1, 1] -> ``[768]`` L2-normalised embedding."""
        T = clip_m1.shape[1]
        assert T >= 2 * self.N_FRAMES - 1, f"ViCLIP protocol needs >= 15 frames, got {T}"
        f = clip_m1[:, 0:2 * self.N_FRAMES:2].permute(1, 0, 2, 3).unsqueeze(0)   # [1, 8, 3, H, W]
        f = (f.to(self.device, torch.float32).clamp(-1, 1) + 1) / 2
        b, t, c, h, w = f.shape
        f = F.interpolate(f.reshape(b * t, c, h, w), size=(224, 224), mode="bilinear",
                          align_corners=False, antialias=True).reshape(b, t, c, 224, 224)
        with self._autocast():
            v = self.model.get_vid_features((f - self.mean) / self.std)
        return F.normalize(v.float().reshape(-1), dim=-1).cpu()


def _stub_viclip_imports():
    """ViCLIP's package imports cv2 (for a numpy preprocessing helper we do not use) and
    pkg_resources (for one version compare). Stub them only when they are genuinely missing."""
    import importlib.machinery
    import importlib.util
    import types

    def missing(name):
        if name in sys.modules:
            return False
        try:
            return importlib.util.find_spec(name) is None
        except (ImportError, ValueError):
            return True

    if missing("pkg_resources"):
        import packaging
        import packaging.version  # noqa: F401
        m = types.ModuleType("pkg_resources")
        m.__spec__ = importlib.machinery.ModuleSpec("pkg_resources", loader=None)
        m.packaging = packaging
        sys.modules["pkg_resources"] = m
    if missing("cv2"):
        m = types.ModuleType("cv2")
        m.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
        m.INTER_LINEAR = 1
        sys.modules["cv2"] = m


class UMTKinetics600:
    """UMT-L/16 fine-tuned on Kinetics-600: first 16 of 17 frames, 224 px, top-1 class.

    The released classifier head is not in alphabetical label order. ``k600_umt_head_order.json``
    (shipped in ``semantok/data``) maps each class name to its head index; it was recovered
    empirically and is what the paper's numbers use. A few similar classes share a head.
    """

    def __init__(self, device, models_dir=None):
        d = models_dir or default_eval_models_dir()
        code = os.path.join(d, "VBench", "vbench", "third_party", "umt", "models", "modeling_finetune.py")
        weights = os.path.join(d, "l16_ptk710_ftk710_ftk600_f16_res224.pth")
        for p in (code, weights):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} missing; run scripts/download_eval_models.sh")
        # Load the one file by path: the package __init__ imports unrelated models and their deps.
        import importlib.util
        spec = importlib.util.spec_from_file_location("_umt_modeling_finetune", code)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.model = mod.vit_large_patch16_224(
            pretrained=False, num_classes=600, all_frames=16, tubelet_size=1,
            use_learnable_pos_emb=False, fc_drop_rate=0., drop_rate=0., drop_path_rate=0.2,
            attn_drop_rate=0., use_checkpoint=False, checkpoint_num=0, use_mean_pooling=True,
            init_scale=0.001)
        ck = torch.load(weights, map_location="cpu")
        for k in ("model", "module", "state_dict"):
            if isinstance(ck, dict) and isinstance(ck.get(k), dict):
                ck = ck[k]
                break
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ck.items()}
        miss, unexp = self.model.load_state_dict(sd, strict=False)
        # The fixed sinusoidal `pos_embed` is a buffer built at init, so it is the only allowed miss.
        if [m for m in miss if m != "pos_embed"] or unexp:
            raise RuntimeError(f"UMT state-dict mismatch: missing={miss[:3]} unexpected={unexp[:3]}")
        self.model = self.model.to(device).eval()
        self.device = device
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "data", "k600_umt_head_order.json")) as f:
            self.head_of = {_norm(k): int(v) for k, v in json.load(f).items()}
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def predict(self, clip_m1):
        x = clip_m1.to(self.device).float()[:, :16]                  # first 16 of 17 frames
        x = x.add(1.0).mul(0.5).clamp(0, 1).permute(1, 0, 2, 3)      # [16, 3, H, W]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        x = ((x - self.mean) / self.std).permute(1, 0, 2, 3).unsqueeze(0)   # [1, 3, 16, 224, 224]
        return int(self.model(x)[0].argmax())

    def hit(self, clip_m1, class_name):
        return float(self.predict(clip_m1) == self.head_of[_norm(class_name)])


def _norm(s):
    return " ".join(s.strip().lower().replace("_", " ").split())


# --------------------------------------------------------------------------- paired pixel metrics ---
def psnr(pred01, target01):
    """PSNR (dB) of one clip ``[3, T, H, W]`` in [0, 1]: one MSE over the whole clip."""
    mse = float(torch.mean((pred01.float() - target01.float()) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def _gaussian_window(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)[None, None]


def ssim(pred01, target01, window_size=11):
    """Mean SSIM of one clip ``[3, T, H, W]`` in [0, 1]: 11x11 Gaussian window (sigma 1.5, zero
    padding), per frame and channel, averaged."""
    C, T, H, W = pred01.shape
    p = pred01.float().reshape(C * T, 1, H, W)
    t = target01.float().reshape(C * T, 1, H, W)
    win = _gaussian_window(window_size).to(p.device, p.dtype)
    pad = window_size // 2
    mu_p, mu_t = F.conv2d(p, win, padding=pad), F.conv2d(t, win, padding=pad)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sig_p = F.conv2d(p * p, win, padding=pad) - mu_p2
    sig_t = F.conv2d(t * t, win, padding=pad) - mu_t2
    sig_pt = F.conv2d(p * t, win, padding=pad) - mu_pt
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / ((mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2))
    return float(smap.mean())
