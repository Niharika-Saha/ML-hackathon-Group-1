# hackman/agents/qlearning.py
from __future__ import annotations
import random
import pickle
from collections import defaultdict
from typing import Dict, Set
import numpy as np

from ..utils import ALPHABET

class TabularQAgent:
    """
    Compact tabular Q over a coarse abstraction.
    State key = (len(mask), lives, top3_letters_by_hmm, guessed_count_bucket)
    """
    def __init__(self, eps_start=0.3, eps_end=0.01, eps_decay_steps=200_000, lr=0.3, gamma=0.95, seed=0):
        self.Q = defaultdict(lambda: np.zeros(26, dtype=np.float32))
        self.steps = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.lr = lr
        self.gamma = gamma
        random.seed(seed)

    @staticmethod
    def _topk_letters(hmm_probs: Dict[str, float], k=3):
        return tuple(sorted(ALPHABET, key=lambda c: hmm_probs.get(c, 0.0), reverse=True)[:k])

    @staticmethod
    def _state_key(mask: str, lives: int, hmm_probs: Dict[str, float], guessed: Set[str]):
        guessed_bucket = min(5, len(guessed)//5)  # 0..5 buckets
        top3 = TabularQAgent._topk_letters(hmm_probs, 3)
        return (len(mask), lives, top3, guessed_bucket)

    def epsilon(self):
        frac = min(1.0, self.steps / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, mask: str, lives: int, hmm_probs: Dict[str, float], guessed: Set[str]):
        self.steps += 1
        s = self._state_key(mask, lives, hmm_probs, guessed)
        eps = self.epsilon()
        if random.random() < eps:
            available = [i for i, ch in enumerate(ALPHABET) if ch not in guessed]
            return random.choice(available)
        q = self.Q[s]
        q_masked = q.copy()
        for i, ch in enumerate(ALPHABET):
            if ch in guessed:
                q_masked[i] = -1e9
        return int(q_masked.argmax())

    def update(self, prev_mask: str, prev_lives: int, prev_hmm: Dict[str, float], prev_guessed: Set[str],
               action_idx: int, reward: float, next_mask: str, next_lives: int, next_hmm: Dict[str, float],
               next_guessed: Set[str], done: bool):
        s = self._state_key(prev_mask, prev_lives, prev_hmm, prev_guessed)
        sp = self._state_key(next_mask, next_lives, next_hmm, next_guessed)
        q = self.Q[s]
        target = reward
        if not done:
            target += self.gamma * self.Q[sp].max()
        q[action_idx] += self.lr * (target - q[action_idx])

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(26, dtype=np.float32))
        for k, v in data.items():
            self.Q[k] = np.array(v, dtype=np.float32)
