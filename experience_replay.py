import random
import typing as tt
from dataclasses import dataclass

import numpy as np


# Single frame: 84x84 = 7,056.
# Grayscale frame requires 1 byte.
# Four stacked frames: 84x84x4 = 28,224 bytes.
# So single transition requires ~56 KB (checked in Jupyter Notebook).
@dataclass
class Experience:
    """A single experience tuple from the environment."""

    state: np.ndarray  # \phi(s_t)
    action: int  # a from A = {1, ..., K}
    reward: float
    is_done: bool
    new_state: np.ndarray  # \phi(s_{t+1})


class ExperienceReplay:
    """Replay mechanism storing experiences in a circular buffer."""

    def __init__(self, size: int) -> None:
        """Initialize the replay memory to its size.

        Args:
            `size`: The maximum number of experiences to store.
        """
        assert size > 0, "Replay buffer size must be positive."
        self.size = size
        self.memory = []  # Underlying list to store transitions.
        self.position = 0  # Pointer to the next insertion point.

    def __len__(self) -> int:
        """Return the current size of the replay memory."""
        return len(self.memory)

    def store(self, experience: Experience) -> None:
        """Store a new experience into the replay memory.

        Args:
            `experience`: The experience to store.
        """
        if len(self.memory) < self.size:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size: int) -> tt.List[Experience]:
        """Sample a batch of experiences from the replay memory.

        Args:
            `batch_size`: The number of experiences to sample.
        """
        assert len(self.memory) >= batch_size, (
            f"Buffer only has {len(self.memory)} elements."
        )
        return random.sample(population=self.memory, k=batch_size)
