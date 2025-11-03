# hackman/features.py
from __future__ import annotations
import math
import numpy as np
from .utils import ALPHABET, entropy_from_probs

MASK_VOCAB = '_' + ALPHABET  # 27 symbols
SYMBOL_TO_IDX = {ch: i for i, ch in enumerate(MASK_VOCAB)}

def _mask_onehot(mask: str, max_len: int = 20) -> np.ndarray:
    L = min(len(mask), max_len)
    arr = np.zeros((max_len, len(MASK_VOCAB)), dtype=np.float32)
    for i in range(L):
        ch = mask[i] if mask[i] in SYMBOL_TO_IDX else '_'
        arr[i, SYMBOL_TO_IDX[ch]] = 1.0
    return arr.reshape(-1)  # (max_len * 27,)

def _guessed_bitmask(guessed: set[str]) -> np.ndarray:
    vec = np.zeros((len(ALPHABET),), dtype=np.float32)
    for i, ch in enumerate(ALPHABET):
        if ch in guessed:
            vec[i] = 1.0
    return vec

def _hmm_probs_vector(hmm_probs: dict[str, float] | None) -> np.ndarray:
    vec = np.zeros((len(ALPHABET),), dtype=np.float32)
    if not hmm_probs:
        vec[:] = 1.0 / len(ALPHABET)
    else:
        total = sum(max(0.0, hmm_probs.get(ch, 0.0)) for ch in ALPHABET)
        if total <= 0:
            vec[:] = 1.0 / len(ALPHABET)
        else:
            for i, ch in enumerate(ALPHABET):
                vec[i] = float(hmm_probs.get(ch, 0.0)) / total
    return vec

def encode_state(mask: str,
                 guessed: set[str],
                 lives_left: int,
                 max_lives: int,
                 hmm_probs: dict[str, float] | None,
                 candidate_count: int | None = None) -> np.ndarray:
    """Return (≈600–650)-D float32 vector."""
    mask_oh = _mask_onehot(mask, max_len=20)
    guessed_bits = _guessed_bitmask(guessed)
    lives_norm = np.array([lives_left / max_lives], dtype=np.float32)
    hmm_vec = _hmm_probs_vector(hmm_probs)

    # optional scalars
    cand_log = np.array([0.0 if candidate_count is None else np.log1p(candidate_count)], dtype=np.float32)
    blank_ratio = float(mask.count('_')) / max(1, len(mask))
    mask_ent_val = - (blank_ratio * math.log(max(blank_ratio, 1e-12)) +
                      (1 - blank_ratio) * math.log(max(1 - blank_ratio, 1e-12)))
    mask_ent = np.array([mask_ent_val], dtype=np.float32)
    hmm_ent = np.array([entropy_from_probs(hmm_vec.tolist())], dtype=np.float32)

    return np.concatenate([mask_oh, guessed_bits, lives_norm, hmm_vec, cand_log, mask_ent, hmm_ent]).astype(np.float32)
