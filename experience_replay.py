import random
import typing as tt
from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    """A single experience tuple from the environment."""

    state: np.ndarray
    action: int
    reward: float
    is_done: bool
    new_state: np.ndarray


class ExperienceReplay:
    """Replay mechanism storing experiences in a circular buffer."""

    def __init__(self, size: int) -> None:
        """Initialize the replay memory to its size.

        Args:
            `size`: The maximum number of experiences to store.
        """
        assert size > 0, "Replay buffer size must be positive."
        self.size = size
        self.memory = []
        self.position = 0

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
