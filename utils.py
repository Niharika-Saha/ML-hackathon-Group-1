# hackman/utils.py
from __future__ import annotations
import math
import random
from typing import Set, List

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
ALPHABET_SET = set(ALPHABET)

def seed_everything(seed: int = 0):
    random.seed(seed)
    try:
        import numpy as np  # optional
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # optional
        torch.manual_seed(seed)
    except Exception:
        pass

def apply_guess(word: str, mask: str, guess: str):
    """Return (new_mask, is_correct, newly_revealed_count)."""
    guess = guess.lower()
    assert len(guess) == 1 and guess in ALPHABET_SET
    if len(word) != len(mask):
        raise ValueError("word and mask length mismatch")
    new_mask_chars = list(mask)
    newly = 0
    for i, ch in enumerate(word):
        if ch == guess and mask[i] == '_':
            new_mask_chars[i] = guess
            newly += 1
    new_mask = ''.join(new_mask_chars)
    return new_mask, (newly > 0), newly

def matches_mask(word: str, mask: str, guessed_wrong: Set[str]) -> bool:
    if len(word) != len(mask):
        return False
    if any((gw in word) for gw in guessed_wrong):
        return False
    for wch, mch in zip(word, mask):
        if mch == '_' and wch in guessed_wrong:
            return False
        if mch != '_' and wch != mch:
            return False
    return True

def entropy_from_probs(probs: List[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(max(p, eps), 2) for p in probs if p > 0)
