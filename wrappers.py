import gymnasium as gym
import highway_env

gym.register_envs(highway_env)


def make_env(env_name: str, m: int, **kwargs) -> gym.Env:
    config = {
        "action": {"type": "DiscreteMetaAction"},
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),  # (W, H)
            "stack_size": m,
            # The RGB weights for the grayscale conversion
            "weights": [0.2989, 0.5870, 0.1140],
            # For graphics, not observation
            "scaling": 1.75,
        },
    }
    env = gym.make(env_name, config=config, **kwargs)
    return env
