# hackman/agents/greedy_hmm.py
from __future__ import annotations
from typing import Set, Dict
from ..utils import ALPHABET

def choose_action(hmm_probs: Dict[str, float], guessed: Set[str]) -> str:
    best = None
    best_p = -1.0
    for ch in ALPHABET:
        if ch in guessed:
            continue
        p = hmm_probs.get(ch, 0.0)
        if p > best_p:
            best_p = p
            best = ch
    return best or 'e'  # fallback
