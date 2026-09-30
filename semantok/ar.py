"""Autoregressive Transformer over SemanTok / VideoFlexTok token ids (inference only).

A LLaMA-style causal decoder: RMSNorm, SwiGLU, learned positional embeddings. Model size is set by
one number, the depth d: width 64d, d heads, SwiGLU hidden 8d/3 rounded up to 64.

    d = 10, 12, 16, 20, 24, 30, 36  ->  49M, 85M, 201M, 393M, 679M, 1.33B, 2.29B non-embedding params

Conditioning:
  * class-to-video (Kinetics-600): a learned class embedding added to the [SOS] input. Index
    ``n_classes`` is the null class, used as the unconditional branch of classifier-free guidance.
  * text-to-video (uCO3D): cross-attention in every block over frozen umT5 caption embeddings. The
    unconditional branch is a learned null context.

The sequence is the time-first flattening of the ``[T, K]`` token grid (see ``semantok.tokens``):
flat position ``i`` holds register ``i // T`` of latent frame ``i % T``. A budget of k tokens per
frame is therefore the first ``T * k`` positions, so one model serves every budget.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu_hidden(dim):
    h = int(8 * dim / 3)
    return max(64, ((h + 63) // 64) * 64)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        d = x.float()
        d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + self.eps)
        return (d * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SelfAttention(nn.Module):
    """Causal self-attention with a KV cache for incremental decoding."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"
        self.h, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cache, start_pos):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, L, self.h, self.hd).transpose(1, 2) for t in (q, k, v))
        cache["k"][:, :, start_pos:start_pos + L] = k
        cache["v"][:, :, start_pos:start_pos + L] = v
        k = cache["k"][:, :, :start_pos + L]
        v = cache["v"][:, :, :start_pos + L]
        # Every cached position is in the past, so a single query step needs no mask.
        o = F.scaled_dot_product_attention(q, k, v, is_causal=L > 1)
        return self.proj(o.transpose(1, 2).reshape(B, L, -1))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, ctx_dim):
        super().__init__()
        self.h, self.hd = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, ctx, ctx_mask=None):
        B, L, _ = x.shape
        S = ctx.shape[1]
        q = self.q(x).view(B, L, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k, v = (t.view(B, S, self.h, self.hd).transpose(1, 2) for t in (k, v))
        m = None if ctx_mask is None else ctx_mask.view(B, 1, 1, S).to(torch.bool)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.proj(o.transpose(1, 2).reshape(B, L, -1))


class Block(nn.Module):
    def __init__(self, dim, heads, ctx_dim=None):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.xattn = None
        if ctx_dim:
            self.nx = RMSNorm(dim)
            self.xattn = CrossAttention(dim, heads, ctx_dim)
        self.n2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, swiglu_hidden(dim))

    def forward(self, x, ctx, ctx_mask, cache, start_pos):
        x = x + self.attn(self.n1(x), cache, start_pos)
        if self.xattn is not None and ctx is not None:
            x = x + self.xattn(self.nx(x), ctx, ctx_mask)
        return x + self.ffn(self.n2(x))


class SemanTokAR(nn.Module):
    """Conditional AR Transformer over a flat token sequence of length ``seq_len``."""

    def __init__(self, depth=16, vocab=64000, seq_len=1280, n_classes=0, ctx_dim=0, n_ctx_null=16,
                 head_bias=False):
        super().__init__()
        dim = 64 * depth
        self.dim, self.depth, self.heads = dim, depth, depth
        self.vocab, self.seq_len, self.n_classes, self.ctx_dim = vocab, seq_len, n_classes, ctx_dim

        self.tok_emb = nn.Embedding(vocab, dim)
        # +1 position for the [SOS] step, which is where the conditioning enters.
        self.pos_emb = nn.Parameter(torch.zeros(seq_len + 1, dim))
        self.sos = nn.Parameter(torch.zeros(dim))
        self.cls_emb = nn.Embedding(n_classes + 1, dim) if n_classes else None
        self.ctx_in = nn.Linear(ctx_dim, dim, bias=False) if ctx_dim else None
        self.ctx_null = nn.Parameter(torch.zeros(n_ctx_null, dim)) if ctx_dim else None
        self.blocks = nn.ModuleList([Block(dim, depth, ctx_dim=(dim if ctx_dim else None))
                                     for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=head_bias)

    @classmethod
    def from_config(cls, cfg):
        return cls(depth=cfg["depth"], vocab=cfg["vocab"], seq_len=cfg["seq_len"],
                   n_classes=cfg.get("n_classes", 0), ctx_dim=cfg.get("ctx_dim", 0),
                   n_ctx_null=cfg.get("n_ctx_null", 16), head_bias=cfg.get("head_bias", False))

    def _prefix(self, B, class_id, device, dtype):
        """The [SOS] input for step 0, with the class embedding added."""
        h = self.sos.to(dtype).view(1, 1, -1).expand(B, 1, -1)
        if self.cls_emb is not None:
            cid = (torch.full((B,), self.n_classes, device=device, dtype=torch.long)
                   if class_id is None else class_id.to(device).long())
            h = h + self.cls_emb(cid).unsqueeze(1).to(dtype)
        return h

    def _context(self, B, text_emb, text_mask, device, dtype):
        """Project the frozen text embedding into the cross-attention space; None -> learned null."""
        if self.ctx_in is None:
            return None, None
        if text_emb is None:
            ctx = self.ctx_null.to(dtype).unsqueeze(0).expand(B, -1, -1)
            return ctx, torch.ones(B, ctx.shape[1], device=device, dtype=torch.bool)
        ctx = self.ctx_in(text_emb.to(dtype))
        if text_mask is None:
            text_mask = torch.ones(ctx.shape[:2], device=device, dtype=torch.bool)
        # A fully masked row (no caption, or the unconditional CFG branch) would make cross-attention
        # read an all -inf row, so those rows fall back to the learned null context.
        empty = ~text_mask.any(dim=1)
        if bool(empty.any()):
            nul = self.ctx_null.to(ctx.dtype)
            n = min(nul.shape[0], ctx.shape[1])
            ctx = ctx.clone()
            ctx[empty, :n] = nul[:n]
            text_mask = text_mask.clone()
            text_mask[empty] = False
            text_mask[empty, :n] = True
        return ctx, text_mask

    @torch.no_grad()
    def sample(self, n, length=None, class_id=None, text_emb=None, text_mask=None,
               temperature=1.0, top_k=0, top_p=1.0, cfg=1.0, device="cuda", generator=None,
               dtype=torch.bfloat16):
        """Sample ``length`` tokens (default: the full sequence) with a KV cache and CFG.

        With ``cfg != 1`` the conditional and unconditional branches run as one batch of 2n; the
        unconditional branch uses the null class / learned null text context.
        Returns long ids ``[n, length]`` in time-first order.
        """
        L = length or self.seq_len
        assert L <= self.seq_len
        guided = cfg != 1.0
        Bx = 2 * n if guided else n

        cid = None
        if class_id is not None:
            class_id = class_id.to(device).long()
            cid = torch.cat([class_id, torch.full_like(class_id, self.n_classes)]) if guided else class_id
        if text_emb is not None:
            text_emb = text_emb.to(device)
            if text_mask is None:
                text_mask = torch.ones(text_emb.shape[:2], device=device, dtype=torch.bool)
            text_mask = text_mask.to(device)
            if guided:
                text_emb = torch.cat([text_emb, text_emb])
                text_mask = torch.cat([text_mask, torch.zeros_like(text_mask)])

        ctx, cmask = self._context(Bx, text_emb, text_mask, device, dtype)
        caches = [{"k": torch.zeros(Bx, self.heads, L, self.dim // self.heads, device=device, dtype=dtype),
                   "v": torch.zeros(Bx, self.heads, L, self.dim // self.heads, device=device, dtype=dtype)}
                  for _ in range(self.depth)]
        out = torch.zeros(n, L, dtype=torch.long, device=device)
        h = self._prefix(Bx, cid, device, dtype)
        for i in range(L):
            x = h + self.pos_emb[i].to(dtype).view(1, 1, -1)
            for blk, c in zip(self.blocks, caches):
                x = blk(x, ctx, cmask, c, i)
            logits = self.head(self.norm(x))[:, -1].float()
            if guided:
                lc, lu = logits[:n], logits[n:]
                logits = lu + cfg * (lc - lu)
            nxt = sample_logits(logits, temperature, top_k, top_p, generator)
            out[:, i] = nxt
            h = self.tok_emb(torch.cat([nxt, nxt]) if guided else nxt).unsqueeze(1).to(dtype)
        return out


def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k and top_k < logits.shape[-1]:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        s, idx = logits.sort(dim=-1, descending=True)
        cum = s.softmax(-1).cumsum(-1)
        # Keep the first token that crosses the threshold, so top_p never yields an empty set.
        s = s.masked_fill((cum - s.softmax(-1)) > top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, idx, s)
    return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)


def load_ar(ref, device="cuda"):
    """Load an AR checkpoint (local directory or ``hf://...``). Returns ``(model, config)``."""
    from . import hub

    cfg, sd = hub.load(ref)
    model = SemanTokAR.from_config(cfg)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval().requires_grad_(False), cfg
