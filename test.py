import argparse

import gymnasium as gym
import highway_env
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from model import DQN
from wrappers import make_env

gym.register_envs(highway_env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="The path to the model to load.",
    )
    parser.add_argument(
        "-r",
        "--record",
        default="videos/testing",
        help="The path to the directory to record the episode, default=videos.",
    )
    args = parser.parse_args()
    env = make_env("highway-v0", m=4, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=args.record,
        name_prefix="test",
        episode_trigger=lambda episode_id: episode_id % 10 == 0,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=5_000)
    net = DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(
        torch.load(
            args.model,
            map_location=lambda storage, _: storage,
            weights_only=True,
        )
    )
    net.eval()
    total_steps = 0
    episode_count = 0
    total_collisions = 0
    returns = []
    episode_lengths = []
    episode_times = []
    while total_steps < 10_000:
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_collision = info.get("crashed", False)
        done = episode_collision
        while not done:
            state_v = torch.tensor(np.expand_dims(obs, axis=0))
            with torch.no_grad():
                q_vals = net(state_v).data.numpy()[0]
            action = int(np.argmax(q_vals))
            obs, reward, is_done, is_trunc, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            # Check for collision
            if info.get("crashed", False):
                episode_collision = True
            done = is_done or is_trunc
        # Episode finished
        episode_info = info.get("episode")
        returns.append(episode_info["r"])
        episode_lengths.append(episode_info["l"])
        episode_times.append(episode_info["t"])
        if episode_collision:
            total_collisions += 1
        # Calculate average return
        avg_return = np.mean(returns)
        episode_count += 1
        print(
            f"Episode {episode_count}: "
            f"Steps: {episode_steps}, "
            f"Return: {episode_reward:.2f}, "
            f"Avg Return: {avg_return:.2f}, "
            f"Time: {episode_info['t']:.2f}s, "
            f"Collision: {episode_collision}, "
            f"Total Steps: {total_steps}"
        )
    # Test completed
    print(f"\nTest completed after {episode_count} episodes, {total_steps} steps")
    print(f"Average return: {np.mean(returns):.4f}")
    print(f"Average return from wrapper: {np.mean(env.return_queue):.4f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f}")
    print(f"Average episode length from wrapper: {np.mean(env.length_queue):.2f}")
    print(f"Average episode time: {np.mean(episode_times):.2f}s")
    print(f"Average episode time from wrapper: {np.mean(env.time_queue):.2f}s")
    print(f"Total collisions: {total_collisions}")
    print(f"Collision rate: {total_collisions / episode_count:.2%}")
    # Show videos
    env.close()
    print(f"\nRecorded videos saved to {args.record}")
