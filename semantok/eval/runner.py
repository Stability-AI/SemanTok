"""The paper's evaluation protocols: AR generation per token budget k, and tokenizer reconstruction.

Reference side. Real clips are passed through the VidTok VAE (encode to the posterior mean, decode)
and the metrics compare against that VAE reconstruction, so the numbers measure the tokenizer and
the AR model rather than the fixed VAE.

Randomness. AR sampling uses one generator per batch of 64, seeded ``seed + start + 1000 k``; the
flow decoder re-seeds with ``seed`` for every batch of 8. The samples therefore depend on those
batch sizes, which are fixed here to the paper's values.
"""
import os

import numpy as np
import torch
from tqdm import tqdm

from ..tokens import T_LAT, unflatten_time_first
from .metrics import frechet_distance, psnr, ssim

SAMPLE_BATCH = 64
DECODE_BATCH = 8

# Paper settings (Sec. 4 / App. F).
K600_AR_CFG = {1: 3.0, 2: 3.0, 4: 3.0, 8: 2.0, 16: 2.0, 32: 2.0, 64: 1.0, 128: 1.0, 256: 1.0}
K600_N_SAMPLES = {1: 4096, 4: 3072}           # other k: 2048
UCO3D_AR_CFG = 3.0
UCO3D_N_SAMPLES = 2560                        # per split (ID and OOD)
UCO3D_MIX = {"id": 1014, "ood": 152}          # official val ratio used to pool the two splits


def to01(x):
    return ((x.float() + 1.0) / 2.0).clamp(0, 1)


class Extractors:
    """The feature extractors shared by every protocol. ``umt`` only for Kinetics-600."""

    def __init__(self, device, models_dir=None, with_umt=False):
        from .metrics import I3DFVD, InceptionFrames, UMTKinetics600, ViCLIP

        self.fid = InceptionFrames(device)
        self.fvd = I3DFVD(device, models_dir)
        self.viclip = ViCLIP(device, models_dir)
        self.umt = UMTKinetics600(device, models_dir) if with_umt else None

    def clip_features(self, clip_m1):
        """One clip in [-1, 1] -> dict of per-clip features."""
        c01 = to01(clip_m1)
        return {"fid": self.fid(c01).cpu(), "fvd": self.fvd(c01).cpu(),
                "viclip": self.viclip.video_features(clip_m1).unsqueeze(0)}


def _batches(pool, idx, batch, workers=8):
    """Load clips ``pool.load(i)`` for i in ``idx`` in order, ``batch`` at a time."""
    ds = [int(i) for i in idx]

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, j):
            return pool.load(ds[j])

    yield from torch.utils.data.DataLoader(_DS(), batch_size=batch, num_workers=workers,
                                           shuffle=False)


def _latent_batches(tok, pool, workers=8):
    """Yield ``(clips, latents)`` in pool order, ``DECODE_BATCH`` clips at a time.

    A pool that defines ``latents(i)`` (precomputed VidTok posterior means) skips video decoding;
    ``clips`` is then None.
    """
    if hasattr(pool, "latents"):
        for b in range(0, len(pool), DECODE_BATCH):
            idx = range(b, min(b + DECODE_BATCH, len(pool)))
            yield None, torch.stack([pool.latents(i) for i in idx]).to(tok.device)
        return
    for clips in _batches(pool, range(len(pool)), DECODE_BATCH, workers):
        yield clips, tok.vae_encode(clips)


def _stack(rows):
    return {k: torch.cat([r[k] for r in rows]) for k in rows[0]}


# ----------------------------------------------------------------------------- reference bank ---
@torch.no_grad()
def reference_features(tok, pool, ex, cache_dir=None, workers=8):
    """Features of the VAE reconstruction of every pool clip, cached per pool."""
    path = os.path.join(cache_dir, f"reference_{pool.name}.pt") if cache_dir else None
    if path and os.path.isfile(path):
        return torch.load(path)
    rows = []
    for _, lat in tqdm(_latent_batches(tok, pool, workers), desc=f"reference {pool.name}",
                       total=-(-len(pool) // DECODE_BATCH)):
        for r in tok.vae_decode(lat):
            rows.append(ex.clip_features(r))
    feats = _stack(rows)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        torch.save(feats, tmp)
        os.replace(tmp, path)          # atomic: parallel runs may share one cache
    return feats


def _scores(ref, gen, ref_rows=None, texts=None, ex=None):
    """Set metrics between reference and generated features; paired ones when rows/texts given."""
    out = {"fvd": frechet_distance(ref["fvd"], gen["fvd"]),
           "fid": frechet_distance(ref["fid"], gen["fid"])}
    if texts is not None:
        tf = ex.viclip.text_features(texts)
        out["viclip"] = float((gen["viclip"] * tf).sum(-1).mean())
    if ref_rows is not None:
        out["clipv"] = float((gen["viclip"] * ref["viclip"][ref_rows]).sum(-1).mean())
    return out


# --------------------------------------------------------------------------------- generation ---
@torch.no_grad()
def sample_tokens(ar, pool, pick, k, cfg, seed, device, text_encoder=None):
    """AR samples for the conditionings of ``pool[pick]`` -> token ids ``[N, 5, k]``."""
    torch.manual_seed(seed)
    out = []
    for s in tqdm(range(0, len(pick), SAMPLE_BATCH), desc=f"AR k={k}"):
        rows = pick[s:s + SAMPLE_BATCH]
        kw = {}
        if pool.dataset == "k600":
            kw["class_id"] = torch.tensor([pool.cond(i) for i in rows], device=device)
        else:
            kw["text_emb"], kw["text_mask"] = text_encoder.embed([pool.cond(i) for i in rows])
        g = torch.Generator(device=device).manual_seed(seed + s + 1000 * k)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = ar.sample(len(rows), length=T_LAT * k, cfg=cfg, device=device, generator=g, **kw)
        out.append(unflatten_time_first(ids).cpu())
    return torch.cat(out)


@torch.no_grad()
def eval_generation(ar, tok, pool, ex, k, n_samples, cfg, reference, seed=0, text_encoder=None,
                    save_videos=None):
    """One (AR model, pool, k) cell. Returns (scores, per-sample features)."""
    pick = np.random.default_rng(seed).choice(len(pool), n_samples, replace=n_samples > len(pool))
    tokens = sample_tokens(ar, pool, pick, k, cfg, seed, tok.device, text_encoder)
    rows, hits = [], []
    for b in tqdm(range(0, len(pick), DECODE_BATCH), desc=f"decode k={k}"):
        videos = tok.decode(tokens[b:b + DECODE_BATCH], k=k, seed=seed)
        for j, v in enumerate(videos):
            rows.append(ex.clip_features(v))
            if ex.umt is not None:
                hits.append(ex.umt.hit(v, pool.text(pick[b + j])))
            if save_videos is not None:
                save_videos(b + j, v)
    gen = _stack(rows)
    scores = _scores(reference, gen, ref_rows=torch.as_tensor(pick),
                     texts=[pool.text(i) for i in pick], ex=ex)
    if hits:
        scores["class_acc"] = float(np.mean(hits))
    scores.update(k=k, n_samples=int(n_samples), ar_cfg=float(cfg))
    return scores, gen


def _subsample(n_total, n_take, seed):
    if n_take >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, size=n_take, replace=False))


def _take(feats, idx, n_clips):
    """Select clip units (FID has one row per frame)."""
    out = {}
    for name, x in feats.items():
        rpc = x.shape[0] // n_clips
        sel = torch.as_tensor(idx)
        out[name] = x.reshape(n_clips, rpc, *x.shape[1:])[sel].reshape(-1, *x.shape[1:])
    return out


def pool_uco3d(split_scores, refs, gens):
    """Pool the uCO3D ID and OOD evaluations at the official val ratio (1014:152).

    FVD and FID are one Frechet distance over the concatenated clips (1014 drawn from the ID side,
    152 from the OOD side, independently for reference and generated sets); ViCLIP and ClipV are
    the ratio-weighted means of the per-split values.
    """
    ref_parts, gen_parts = [], []
    for seed, split in enumerate(("id", "ood")):
        n = UCO3D_MIX[split]
        nr, ng = refs[split]["fvd"].shape[0], gens[split]["fvd"].shape[0]
        ref_parts.append(_take(refs[split], _subsample(nr, min(n, nr, ng), seed), nr))
        gen_parts.append(_take(gens[split], _subsample(ng, min(n, nr, ng), seed), ng))
    ref, gen = ({k: torch.cat([p[k] for p in parts]) for k in parts[0]}
                for parts in (ref_parts, gen_parts))
    w = {s: UCO3D_MIX[s] / sum(UCO3D_MIX.values()) for s in UCO3D_MIX}
    out = {"fvd": frechet_distance(ref["fvd"], gen["fvd"]),
           "fid": frechet_distance(ref["fid"], gen["fid"])}
    for m in ("viclip", "clipv"):
        if all(m in split_scores[s] for s in w):
            out[m] = sum(w[s] * split_scores[s][m] for s in w)
    return out


# ----------------------------------------------------------------------------- reconstruction ---
@torch.no_grad()
def eval_reconstruction(tok, pool, ex, ks, reference=None, seed=0, pixel_only=False, workers=8,
                        save_videos=None, encode_kwargs=None):
    """Encode every pool clip, decode it from its first k tokens, and score it.

    Paired metrics compare each reconstruction with the VAE reconstruction of the same clip: PSNR,
    SSIM and ClipV. Set metrics (FVD, FID) compare against ``reference`` (features of a reference
    pool, possibly a different set of clips); ViCLIP and class accuracy use the clip's own label.
    ``pixel_only`` reports PSNR / SSIM only. ``encode_kwargs(i0, clips)`` may return extra keyword
    arguments for ``tok.encode`` (e.g. a different DINO source).
    """
    per_k = {k: {"rows": [], "psnr": [], "ssim": [], "hits": []} for k in ks}
    target_vic = []
    n = 0
    for clips, lat in tqdm(_latent_batches(tok, pool, workers), desc=f"reconstruct {pool.name}",
                           total=-(-len(pool) // DECODE_BATCH)):
        target = tok.vae_decode(lat)
        extra = encode_kwargs(n, clips) if encode_kwargs else {}
        # Without source clips (latent pools), SemanTok's DINO input is read from the VAE reconstruction.
        tokens = tok.encode(target if clips is None else clips, latents=lat, **extra)
        if not pixel_only:
            target_vic += [ex.viclip.video_features(t) for t in target]
        for k in ks:
            rec = tok.decode(tokens, k=k, seed=seed)
            acc = per_k[k]
            for j, v in enumerate(rec):
                acc["psnr"].append(psnr(to01(v), to01(target[j])))
                acc["ssim"].append(ssim(to01(v), to01(target[j])))
                if not pixel_only:
                    acc["rows"].append(ex.clip_features(v))
                    if ex.umt is not None:
                        acc["hits"].append(ex.umt.hit(v, pool.text(n + j)))
                if save_videos is not None:
                    save_videos(k, n + j, v)
        n += len(lat)
    out, feats = {}, {}
    for k in ks:
        acc = per_k[k]
        s = {"psnr": float(np.mean(acc["psnr"])), "ssim": float(np.mean(acc["ssim"])), "n": n}
        if not pixel_only:
            gen = _stack(acc["rows"])
            feats[k] = gen
            s.update(_scores(reference, gen, texts=[pool.text(i) for i in range(n)], ex=ex))
            s["clipv"] = float((gen["viclip"] * torch.stack(target_vic)).sum(-1).mean())
            if acc["hits"]:
                s["class_acc"] = float(np.mean(acc["hits"]))
        out[k] = s
    return out, feats
