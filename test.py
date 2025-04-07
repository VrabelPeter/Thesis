import argparse

import gymnasium as gym
import highway_env
import numpy as np
import torch

from hyperparameters import parameters
from model import DQN
from wrappers import make_env

gym.register_envs(highway_env)

STEP_LIMIT = 10_000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Test a trained DQN agent on the {parameters['env_name']} environment.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="The path to the model to load.",
    )
    parser.add_argument(
        "-r",
        "--record",
        default="videos/test",
        type=str,
        help="The path to the directory to save the videos (default=`videos/test`).",
    )
    args = parser.parse_args()
    env = make_env(
        parameters["env_name"],
        m=parameters["agent_history_length"],
        video_folder=args.record,
        name_prefix="test",
        record_frequency=10,
        buffer_length=STEP_LIMIT,
        render_mode="rgb_array",
    )
    net = DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(
        torch.load(
            args.model,
            map_location=lambda storage, _: storage,
            weights_only=True,
        )
    )
    total_steps = 0
    episode_count = 0
    total_collisions = 0
    episode_speeds = []
    while total_steps < STEP_LIMIT:
        obs, info = env.reset()
        episode_collision = info["crashed"]
        done = episode_collision
        speed_values = []
        while not done:
            state_v = torch.tensor(np.expand_dims(obs, axis=0))
            with torch.no_grad():
                q_vals = net(state_v).data.numpy()[0]
            action = int(np.argmax(q_vals))
            obs, reward, is_done, is_trunc, info = env.step(action)
            done = is_done or is_trunc
            if info["crashed"]:
                episode_collision = True
            speed_values.append(info["speed"])
            total_steps += 1
        # Episode finished
        episode_info = info.get("episode")
        avg_speed = np.mean(speed_values) if speed_values else 0.0
        episode_speeds.append(avg_speed)
        if episode_collision:
            total_collisions += 1
        episode_count += 1
        print(
            f"Total steps: {total_steps}:",
            f"Episode: {episode_count},",
            f"Steps: {episode_info['l']},",
            f"Return: {episode_info['r']:.2f},",
            f"Time: {episode_info['t']:.2f}s,",
            f"Avg Speed: {avg_speed:.2f},",
            f"Collision: {episode_collision}",
        )
    # Test completed
    print(f"\nTest completed after {episode_count} episodes, {total_steps} steps")
    print(f"Average return: {np.mean(env.return_queue):.2f}")
    print(f"Average episode length: {np.mean(env.length_queue):.2f}")
    print(f"Average episode time: {np.mean(env.time_queue)::.2f}s")
    print(f"Average episode speed: {np.mean(episode_speeds):.2f}")
    print(f"Total collisions: {total_collisions}")
    print(f"Collision rate: {total_collisions / episode_count:.2%}")
    env.close()
    print(f"\nRecorded videos saved to {args.record}")
