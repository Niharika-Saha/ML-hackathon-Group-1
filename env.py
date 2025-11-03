# hackman/env.py
from __future__ import annotations
import random
from typing import Dict, Set, List

from .features import encode_state
from .utils import (
    ALPHABET,
    ALPHABET_SET,
    apply_guess,
    matches_mask,
    seed_everything,
)

class HangmanEnv:
    def __init__(self, words: List[str], lives: int = 6, hmm=None, seed: int = 0):
        self.words = [w.strip().lower() for w in words if w.strip()]
        self.max_lives = lives
        self.hmm = hmm  # expects .letter_posteriors(mask, guessed)
        seed_everything(seed)
        self.rng = random.Random(seed)

        # runtime state
        self.word = None
        self.mask = None
        self.lives = None
        self.guessed: Set[str] = set()
        self.guessed_wrong: Set[str] = set()
        self.total_wrong = 0
        self.total_repeated = 0

    def _candidate_count(self) -> int:
        if self.word is None:
            return 0
        L = len(self.word)
        pool = (w for w in self.words if len(w) == L)
        return sum(1 for w in pool if matches_mask(w, self.mask, self.guessed_wrong))

    def _hmm_probs(self) -> Dict[str, float]:
        if self.hmm is None:
            return {ch: 1.0 / 26.0 for ch in ALPHABET}
        try:
            return self.hmm.letter_posteriors(self.mask, set(self.guessed))
        except Exception:
            return {ch: 1.0 / 26.0 for ch in ALPHABET}

    def _state(self):
        hmm_probs = self._hmm_probs()
        cand_count = self._candidate_count()
        return encode_state(
            mask=self.mask,
            guessed=self.guessed,
            lives_left=self.lives,
            max_lives=self.max_lives,
            hmm_probs=hmm_probs,
            candidate_count=cand_count,
        )

    def reset(self, word: str | None = None):
        self.word = word or self.rng.choice(self.words)
        self.mask = '_' * len(self.word)
        self.lives = self.max_lives
        self.guessed.clear()
        self.guessed_wrong.clear()
        self.total_wrong = 0
        self.total_repeated = 0
        return self._state()

    def step(self, action_letter: str):
        letter = action_letter.lower()
        info = {}
        reward = -0.05  # small step penalty
        done = False

        # invalid action safety
        if len(letter) != 1 or letter not in ALPHABET_SET:
            info["invalid"] = True
            return self._state(), reward - 1.0, False, info

        # repeated guess
        if letter in self.guessed:
            self.total_repeated += 1
            return self._state(), reward - 2.0, False, {"repeated": True}

        # fresh guess
        self.guessed.add(letter)
        new_mask, is_correct, newly = apply_guess(self.word, self.mask, letter)
        if is_correct:
            self.mask = new_mask
            reward += 1.0
            info["correct_newly_revealed"] = newly
            if self.mask == self.word:
                reward += 5.0
                done = True
                info["win"] = True
        else:
            self.guessed_wrong.add(letter)
            self.lives -= 1
            self.total_wrong += 1
            reward -= 1.0
            info["wrong"] = True
            if self.lives <= 0:
                reward -= 3.0
                done = True
                info["lose"] = True

        return self._state(), reward, done, info
