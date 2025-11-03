# hackman/agents/dqn.py
from __future__ import annotations
from typing import Deque, Tuple, Set
from collections import deque
import random
import numpy as np

from ..utils import ALPHABET, seed_everything

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise ImportError(
        "PyTorch is required for DQN. Install with `pip install torch` or skip importing DQNAgent."
    ) from e

class QNet(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128), out_dim: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], out_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNConfig:
    def __init__(self,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 eps_start: float = 0.2,
                 eps_end: float = 0.01,
                 eps_decay_steps: int = 200_000,
                 batch_size: int = 256,
                 buffer_size: int = 50_000,
                 warmup: int = 1_000,
                 target_tau: float = 0.01,
                 seed: int = 0):
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.warmup = warmup
        self.target_tau = target_tau
        self.seed = seed

class DQNAgent:
    def __init__(self, state_dim: int, cfg: DQNConfig = DQNConfig()):
        seed_everything(cfg.seed)
        self.cfg = cfg
        self.policy = QNet(state_dim)
        self.target = QNet(state_dim)
        self.target.load_state_dict(self.policy.state_dict())
        self.optim = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=cfg.buffer_size)
        self.steps = 0
        self.loss_fn = nn.SmoothL1Loss()

    def epsilon(self):
        frac = min(1.0, self.steps / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, state_vec: np.ndarray, guessed: Set[str]):
        self.steps += 1
        eps = self.epsilon()
        if random.random() < eps:
            available = [i for i, ch in enumerate(ALPHABET) if ch not in guessed]
            return random.choice(available)
        with torch.no_grad():
            q = self.policy(torch.from_numpy(state_vec).float().unsqueeze(0)).squeeze(0)
            q = q.numpy().copy()
            for i, ch in enumerate(ALPHABET):
                if ch in guessed:
                    q[i] = -1e9
            return int(q.argmax())

    def push(self, s, a, r, sp, done):
        self.buffer.append((s, a, r, sp, done))

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.policy.parameters(), self.target.parameters()):
                tp.data.copy_(self.cfg.target_tau * p.data + (1 - self.cfg.target_tau) * tp.data)

    def train_step(self):
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.warmup):
            return 0.0
        import numpy as np
        import torch
        batch = random.sample(self.buffer, self.cfg.batch_size)
        s, a, r, sp, d = zip(*batch)
        s = torch.from_numpy(np.stack(s)).float()
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float32)
        sp = torch.from_numpy(np.stack(sp)).float()
        d = torch.tensor(d, dtype=torch.float32)

        q = self.policy(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(sp).max(1)[0]
            target = r + self.cfg.gamma * (1 - d) * q_next
        loss = self.loss_fn(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.soft_update()
        return float(loss.item())
