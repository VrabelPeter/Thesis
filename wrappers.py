from typing import Optional

import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3.common.monitor import Monitor

gym.register_envs(highway_env)


def make_base_env(
    env_name: str,
    m: int,
    **kwargs,
) -> gym.Env:
    """Create a base environment with the specified name and parameters.

    Args:
        env_name (str): The name of the environment to create.
        m (int): The number of frames to stack.
        **kwargs: Additional keyword arguments to pass to the environment.

    Returns:
        gym.Env: The created environment.
    """
    config = {
        "action": {
            "type": "DiscreteMetaAction",
        },
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),  # (W, H)
            "stack_size": m,
            # The RGB weights for the grayscale conversion
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75,  # For graphics, not observation
        },
    }
    return gym.make(env_name, config=config, **kwargs)


def wrap_env_with_recorder(
    env: gym.Env,
    name_prefix: str,
    video_folder: Optional[str] = None,
    record_frequency: int = 0,
):
    """Wrap the environment with video recording capabilities.

    Args:
        env (gym.Env): The environment to wrap.
        name_prefix (str): Prefix for the video file names.
        video_folder (Optional[str]): Folder to save videos. If None, no videos are recorded.
        record_frequency (int): Frequency of recording episodes.

    Returns:
        gym.Env: The wrapped environment with video recording capabilities.
    """
    if video_folder is not None and record_frequency > 0:
        print(f"Recording videos to '{video_folder}'")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            lambda episode: episode % record_frequency == 0,
            name_prefix=name_prefix,
        )
    return env


class CustomMetricsWrapper(gym.Wrapper):
    """
    A wrapper to extract and calculate custom metrics from the highway-env environment.

    This wrapper calculates the mean speed per episode and tracks whether an episode
    ended in a crash. It adds this information to the `info` dictionary upon episode end.
    """

    def __init__(self, env):
        super().__init__(env)
        self.episode_speeds: list[float] = []

    def reset(self, **kwargs):
        self.episode_speeds = []
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_speeds.append(info.get("speed", 0))
        if terminated or truncated:  # End of episode
            # Add custom metrics to the info dict for the `Monitor` wrapper
            info["episode_metrics"] = {
                "mean_speed": np.mean(self.episode_speeds)
                if self.episode_speeds
                else 0,
            }
        return obs, reward, terminated, truncated, info


def create_sb3_env(
    env_name: str,
    m: int,
    **kwargs,
) -> gym.Env:
    """Create environment compatible with Stable Baselines 3."""
    env = make_base_env(env_name, m, **kwargs)
    env = CustomMetricsWrapper(env)
    # Some logging values (like `ep_rew_mean`, `ep_len_mean`) are only available when using a `Monitor` wrapper - see notes
    env = Monitor(
        env,
        info_keywords=("episode_metrics",),
    )
    return env
