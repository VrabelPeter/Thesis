import gymnasium as gym
import highway_env

gym.register_envs(highway_env)


def make_env(
    env_name: str,
    m: int,
    video_folder: str,
    name_prefix: str,
    record_frequency: int,
    buffer_length: int,
    **kwargs,
) -> gym.Env:
    """Create a gym environment with the specified name and parameters.

    Args:
        env_name (str): The name of the environment to create.
        m (int): The number of frames to stack.
        video_folder (str): Folder to save videos.
        name_prefix (str): Prefix for video file names.
        record_frequency (int): How often to record videos.
        buffer_length (int): Length of the buffer for episode statistics.
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
    env = gym.make(env_name, config=config, **kwargs)
    if video_folder is not None and record_frequency > 0:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            lambda episode: episode % record_frequency == 0,
            name_prefix=name_prefix,
        )
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length)
    return env
