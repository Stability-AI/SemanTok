"""Token-grid layout shared by the tokenizer and the AR model.

The tokenizer emits ``[T, K]`` ids per clip (T=5 latent frames, K=256 registers per frame). The AR
model reads them TIME-FIRST: register-major, frame-minor, so flat position ``i`` holds register
``i // T`` of frame ``i % T``. The first ``T * k`` positions are then exactly a k-token budget for
every frame, which is why a budget is a prefix of the AR sequence.
"""
T_LAT = 5
N_REG = 256
FSQ_LEVELS = [8, 8, 8, 5, 5, 5]
VOCAB = 64000  # prod(FSQ_LEVELS)


def flatten_time_first(ids):
    """``[..., T, K]`` -> ``[..., K*T]``."""
    T, K = ids.shape[-2], ids.shape[-1]
    return ids.transpose(-2, -1).reshape(*ids.shape[:-2], K * T)


def unflatten_time_first(ids, T=T_LAT):
    """Inverse of ``flatten_time_first``; a prefix of length ``T*k`` returns ``[..., T, k]``."""
    L = ids.shape[-1]
    assert L % T == 0, f"flat length {L} is not a multiple of T={T}"
    return ids.reshape(*ids.shape[:-1], L // T, T).transpose(-2, -1)
