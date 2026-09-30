"""Resolve a checkpoint reference to a local directory holding ``config.json`` + ``model.safetensors``.

A reference is either a local directory or ``hf://<org>/<repo>[@<revision>][/<subdir>]``.
"""
import json
import os


def resolve(ref):
    if not ref.startswith("hf://"):
        if not os.path.isdir(ref):
            raise FileNotFoundError(f"checkpoint directory not found: {ref}")
        return ref
    from huggingface_hub import snapshot_download

    spec = ref[len("hf://"):]
    parts = spec.split("/")
    if len(parts) < 2:
        raise ValueError(f"expected hf://<org>/<repo>[@rev][/subdir], got {ref}")
    repo, revision = "/".join(parts[:2]), None
    if "@" in repo:
        repo, revision = repo.split("@", 1)
    subdir = "/".join(parts[2:])
    patterns = [f"{subdir}/*"] if subdir else None
    root = snapshot_download(repo, revision=revision, allow_patterns=patterns)
    return os.path.join(root, subdir) if subdir else root


def load(ref):
    """Return ``(config_dict, state_dict)`` for a checkpoint reference."""
    from safetensors.torch import load_file

    d = resolve(ref)
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    return cfg, load_file(os.path.join(d, "model.safetensors"))


def join(root, *parts):
    """Child reference of a checkpoint root (local directory or ``hf://org/repo[@rev]``)."""
    return "/".join([root.rstrip("/"), *parts]) if root.startswith("hf://") else os.path.join(root, *parts)
