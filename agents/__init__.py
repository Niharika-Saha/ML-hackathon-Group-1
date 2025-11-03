# hackman/agents/__init__.py
from .greedy_hmm import choose_action as greedy_choose_action
from .qlearning import TabularQAgent
try:
    from .dqn import DQNAgent, DQNConfig
except Exception:
    # torch optional; importing DQN may fail if torch missing
    DQNAgent = None
    DQNConfig = None

__all__ = ["greedy_choose_action", "TabularQAgent", "DQNAgent", "DQNConfig"]
