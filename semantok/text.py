"""umT5-XXL caption embeddings for the text-conditioned (uCO3D) AR models.

The AR models were trained on the umT5-XXL encoder that ships with Wan 2.1. Only its tokenizer and
text encoder are loaded here (``text_encoder/`` and ``tokenizer/`` of the HF repo below); no video
model is involved. The prompt handling reproduces the recipe the training embeddings were made with:
ftfy/HTML clean-up and whitespace collapse, padding to 512 tokens, bf16 encoder, and the padded
positions set to zero. The AR model then reads the first ``max_text_len`` (128) real tokens.
"""
import html
import re

import torch

UMT5_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MAX_SEQUENCE_LENGTH = 512
MAX_TEXT_LEN = 128


def _clean(text):
    import ftfy

    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


class UMT5TextEncoder:
    def __init__(self, device="cuda", repo=UMT5_REPO, revision=None, dtype=torch.bfloat16):
        from transformers import AutoTokenizer, UMT5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer", revision=revision)
        self.model = UMT5EncoderModel.from_pretrained(repo, subfolder="text_encoder",
                                                      revision=revision, torch_dtype=dtype)
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def embed(self, captions, max_text_len=MAX_TEXT_LEN):
        """Captions -> (``[B, L, 4096]`` float32 embeddings, ``[B, L]`` bool mask), L <= max_text_len.

        Embeddings go through a float16 round trip, which is how the training embeddings were
        stored.
        """
        tok = self.tokenizer([_clean(c) for c in captions], padding="max_length",
                             max_length=MAX_SEQUENCE_LENGTH, truncation=True,
                             add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        ids, mask = tok.input_ids.to(self.device), tok.attention_mask.to(self.device)
        emb = self.model(ids, mask).last_hidden_state.to(self.dtype)
        lens = mask.gt(0).sum(1).clamp(max=max_text_len)
        L = int(lens.max())
        out = torch.zeros(len(captions), L, emb.shape[-1], dtype=torch.float32, device=self.device)
        m = torch.zeros(len(captions), L, dtype=torch.bool, device=self.device)
        for i, n in enumerate(lens.tolist()):
            out[i, :n] = emb[i, :n].to(torch.float16).float()
            m[i, :n] = out[i, :n].abs().sum(-1) > 0
        return out, m
