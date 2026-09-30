"""SemanTok / VideoFlexTok tokenizers: 17x128x128 RGB clip <-> [5, 256] FSQ token ids.

Both arms share the architecture of the public VideoFlexTok d18/d18 Kinetics model (VidTok VAE,
18-layer encoder and time-causal flow decoder, FSQ levels [8,8,8,5,5,5] = 64000 ids, 256 nested
registers per latent frame). The modules come unmodified from the upstream ``videoflextok`` package;
a checkpoint's ``config.json`` carries the full module config.

SemanTok differs only in its encoder input:
  * the encoder reads DINOv2-L patch grids concatenated with the VidTok latent (1024 + 16 channels);
  * a projection of a DINOv2 CLS vector is added to register 0 of every latent frame.
Its decoder is the same kind as VideoFlexTok's, so AR generation needs no DINO at all.
"""
from contextlib import nullcontext

import torch
import torch.nn as nn

from . import hub
from .tokens import N_REG

GLOBAL_VECS_KEY = "global_vecs"
DINO_INPUT_KEY = "enc_dino_vae_bthwc"


class GlobalTokenInput(nn.Module):
    """Adds ``proj(vec)`` to register 0. ``vec`` is ``[T, D]`` (one per latent frame) or ``[D]``."""

    def __init__(self, global_dim, enc_dim):
        super().__init__()
        self.proj = nn.Linear(global_dim, enc_dim)

    def forward(self, registers_list, global_vecs):
        out = []
        for reg, gv in zip(registers_list, global_vecs):
            T = reg.shape[1]
            gv = gv.to(self.proj.weight.device, self.proj.weight.dtype)
            delta = self.proj(gv)
            delta = delta.view(1, T, 1, -1) if gv.dim() == 2 else delta.view(1, 1, 1, -1)
            onehot = torch.zeros(reg.shape[-2], device=reg.device, dtype=reg.dtype)
            onehot[0] = 1.0
            out.append(reg + delta.to(reg.dtype) * onehot.view(1, 1, -1, 1))
        return out


def _attach_global_token(vt, global_dim, enc_dim):
    vt.global_token_input = GlobalTokenInput(global_dim, enc_dim)
    reg_mod = vt.encoder.module_dict["enc_register_module"]
    orig_forward, key = reg_mod.forward, reg_mod.registers_write_key

    def forward(data_dict):
        data_dict = orig_forward(data_dict)
        data_dict[key] = vt.global_token_input(data_dict[key], data_dict[GLOBAL_VECS_KEY])
        return data_dict

    reg_mod.forward = forward


def _hub_vae_state(repo, revision):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sd = load_file(hf_hub_download(repo, "model.safetensors", revision=revision))
    p = "video_tokenizer."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p + "vae.")}


class Tokenizer:
    """Load with ``Tokenizer("hf://org/repo/subdir")`` or ``Tokenizer("/path/to/checkpoint_dir")``."""

    def __init__(self, ref, device="cuda", bf16=True):
        from videoflextok.wrappers import VideoFlexTokFromHub

        cfg, sd = hub.load(ref)
        self.meta = cfg["semantok"]
        self.device = device
        self.bf16 = bf16
        wrapper = VideoFlexTokFromHub(cfg["videoflextok"])
        self.video_sizes = tuple(wrapper.vae_video_sizes)
        vt = wrapper.video_tokenizer
        if self.meta["global_token"]:
            enc_dim = cfg["videoflextok"]["video_flex_tok"]["encoder"]["module_dict"][
                "enc_register_module"]["dim"]
            _attach_global_token(vt, global_dim=1024, enc_dim=enc_dim)
        if not any(k.startswith("vae.") for k in sd):
            sd.update(_hub_vae_state(cfg["hub_vae"]["repo"], cfg["hub_vae"]["revision"]))
        vt.load_state_dict(sd, strict=True)
        self.vt = vt.to(device).eval().requires_grad_(False)
        self._dino = None

    @property
    def uses_dino(self):
        return bool(self.meta["dino_input"] or self.meta["global_token"])

    def _autocast(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if self.bf16 and str(self.device) != "cpu" else nullcontext())

    @property
    def dino(self):
        if self._dino is None:
            from .dino import DinoV2
            self._dino = DinoV2(self.device)
        return self._dino

    # ------------------------------------------------------------------------------------- VAE ---
    @torch.no_grad()
    def vae_encode(self, clips):
        """``[B, 3, 17, 128, 128]`` in [-1, 1] -> posterior-mean latents ``[B, 16, 5, 16, 16]``.

        The mean, not a posterior sample, is what the tokenizers were trained and scored on. It goes
        through a float16 round trip, as the training latents were stored in float16.
        """
        with self._autocast():
            moments = self.vt.vae.vidtok.encoder(clips.to(self.device))
        return moments[:, :16].float().half().float()

    @torch.no_grad()
    def vae_decode(self, latents):
        """Latents ``[B, 16, 5, 16, 16]`` -> RGB ``[B, 3, 17, 128, 128]`` in [-1, 1]."""
        vae = self.vt.vae
        dd = vae.decode({vae.vae_latents_read_key: list(latents.to(self.device).split(1))})
        return torch.cat(dd[vae.images_reconst_write_key]).float()

    # ---------------------------------------------------------------------------------- encode ---
    @torch.no_grad()
    def encode(self, clips, dino_clips=None, latents=None, dino_grids=None, global_vecs=None):
        """RGB clips -> token ids ``[B, 5, 256]``.

        ``clips``: ``[B, 3, 17, 128, 128]`` in [-1, 1], the VAE input (and, by default, the source
        of SemanTok's DINO input).
        ``dino_clips``: optional ``[B, 3, 17, H, W]`` source for the DINO input instead (any square
        size; frames are resized to 224).
        ``latents`` skips the VAE encode; ``dino_grids`` (``[B, 5, 16, 16, 1024]``) and
        ``global_vecs`` (``[B, 5, 1024]`` or ``[B, 1024]``) skip the DINO passes.
        """
        if latents is None:
            latents = self.vae_encode(clips)
        lat = [l.unsqueeze(0) for l in latents.to(self.device)]
        dd = {"vae_latents": lat}
        src = clips if dino_clips is None else dino_clips
        if self.meta["dino_input"]:
            if dino_grids is None:
                dino_grids = [self.dino.encoder_input(s, agg=self.meta["dino_frame_agg"]) for s in src]
            dd[DINO_INPUT_KEY] = [torch.cat([g.to(self.device).float().unsqueeze(0),
                                             l.permute(0, 2, 3, 4, 1)], dim=-1)
                                  for g, l in zip(dino_grids, lat)]
        if self.meta["global_token"]:
            if global_vecs is None:
                global_vecs = [self.dino.cls(s, mode=self.meta["global_token_mode"]) for s in src]
            dd[GLOBAL_VECS_KEY] = [v.to(self.device).float() for v in global_vecs]
        vt = self.vt
        dd = vt.encoder(dd)
        dd = vt.regularizer(dd)
        return torch.cat([t.reshape(1, *t.shape[-2:]) for t in dd[vt.token_write_key]]).long()

    # ---------------------------------------------------------------------------------- decode ---
    @torch.no_grad()
    def decode(self, tokens, k=None, timesteps=50, guidance_scale=3.0, seed=0):
        """Token ids ``[B, 5, k]`` (or ``[B, 5, 256]`` with a budget ``k``) -> RGB ``[B, 3, 17, 128, 128]``.

        Registers ``k..255`` are replaced by the decoder's learned mask token. The flow sampler's
        noise comes from one generator seeded with ``seed`` per call, so the samples depend on how
        clips are batched; the paper's evaluation decodes batches of 8.
        """
        vt = self.vt
        tokens = tokens.to(self.device)
        k = tokens.shape[-1] if k is None else int(k)
        B, T = tokens.shape[:2]
        quant = vt.regularizer.indices_to_embedding(tokens[:, :, :k])            # [B, T, k, 6]
        codes = torch.zeros(B, T, N_REG, quant.shape[-1], device=self.device, dtype=quant.dtype)
        codes[:, :, :k] = quant
        nd = vt.decoder.module_dict["dec_nested_dropout"]
        dd = {vt.quants_write_key: list(codes.split(1)), nd.eval_keep_k_read_key: [k] * B}
        with self._autocast():
            dd = vt.decode(dd, video_sizes=[self.video_sizes] * B, timesteps=timesteps,
                           guidance_scale=guidance_scale, perform_norm_guidance=True,
                           generator=torch.Generator(self.device).manual_seed(seed),
                           eta=0.0, momentum=0.0, norm_threshold=0.6, verbose=False)
        return torch.cat(dd[vt.image_write_key]).float()
