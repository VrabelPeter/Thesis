import argparse
import collections
import os

import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import DQN

from hyperparameters import parameters
from wrappers import make_base_env, wrap_env_with_recorder

gym.register_envs(highway_env)


def evaluate_sb3_model(
    model_path,
    num_episodes=100,
    record_videos=False,
    video_folder="videos/sb3_eval",
    use_random_agent=False,
):
    """
    Evaluate a Stable Baselines3 DQN model and collect statistics.

    Args:
        model_path: Path to the saved SB3 model (.zip file)
        num_episodes: Number of episodes to evaluate
        record_videos: Whether to record videos of evaluation
        video_folder: Folder to save videos if recording
        use_random_agent: If True, use a random agent instead of the trained model
    """

    if not use_random_agent:
        print(f"Evaluating SB3 model: {model_path}")
    print(f"Number of episodes: {num_episodes}")
    print("=" * 60)

    # Create environment
    env = make_base_env(
        parameters["env_name"],
        m=parameters["agent_history_length"],
        render_mode="rgb_array" if record_videos else None,
    )
    env = wrap_env_with_recorder(
        env,
        name_prefix="sb3_eval",
        video_folder=video_folder if record_videos else None,
        record_frequency=1 if record_videos else 0,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    model = None
    if not use_random_agent:
        # Load the trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = DQN.load(model_path, env=env)
        print(f"Model loaded successfully from {model_path}")

    # Statistics tracking (same as test.py)
    episode_count = 0
    total_collisions = 0
    episode_speeds = []
    total_action_counts = collections.defaultdict(int)
    all_episode_returns = []
    all_episode_lengths = []
    all_episode_times = []

    print("\nStarting evaluation...")
    print("-" * 60)

    for ep_i in range(num_episodes):
        obs, info = env.reset(seed=ep_i)
        episode_collision = info["crashed"]
        done = episode_collision
        speed_values = []
        episode_action_counts = collections.defaultdict(int)
        current_episode_steps = 0
        episode_return = 0.0

        while not done:
            if use_random_agent:
                action = env.action_space.sample()
            else:
                # Use the trained model to predict action
                action_arr, _states = model.predict(obs, deterministic=True)
                action = action_arr.item()

            episode_action_counts[action] += 1
            total_action_counts[action] += 1

            obs, reward, is_done, is_trunc, info = env.step(action)
            done = is_done or is_trunc

            if info["crashed"]:
                episode_collision = True

            speed_values.append(info["speed"])
            episode_return += float(reward)
            current_episode_steps += 1

        # Episode finished - collect statistics
        episode_info = info.get(
            "episode",
            {  # Handle potential missing info if truncated early
                "l": current_episode_steps,
                "r": episode_return,
                "t": np.nan,
            },
        )

        avg_speed = np.mean(speed_values) if speed_values else 0.0
        episode_speeds.append(avg_speed)
        all_episode_returns.append(episode_info.get("r", episode_return))
        all_episode_lengths.append(episode_info.get("l", current_episode_steps))
        all_episode_times.append(episode_info.get("t", np.nan))

        if episode_collision:
            total_collisions += 1

        episode_count += 1

        # Print episode statistics
        action_counts_str = ", ".join(
            [f"Action {k}: {v}" for k, v in sorted(episode_action_counts.items())]
        )
        print(
            f"Episode: {episode_count}/{num_episodes},",
            f"Steps: {episode_info.get('l', current_episode_steps)},",
            f"Return: {episode_info.get('r', episode_return):.2f},",
            f"Time: {episode_info.get('t', np.nan):.2f}s,",
            f"Avg Speed: {avg_speed:.2f} m/s,",
            f"Collision: {episode_collision},",
            f"Action Counts: [{action_counts_str}]",
        )

    env.close()

    # Print final statistics (same format as test.py)
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Evaluation completed after {episode_count} episodes")
    print(f"Average return: {np.mean(all_episode_returns):.2f}")
    print(f"Average episode length: {np.mean(all_episode_lengths):.2f}")

    # Handle episode times (might have NaN values)
    valid_times = [t for t in all_episode_times if not np.isnan(t)]
    if valid_times:
        print(f"Average episode time: {np.mean(valid_times):.2f}s")
    else:
        print("Average episode time: N/A")

    print(f"Average episode speed: {np.mean(episode_speeds):.2f}")
    print(f"Total collisions: {total_collisions}")
    print(f"Collision rate: {total_collisions / episode_count:.2%}")
    print(f"Success rate: {(episode_count - total_collisions) / episode_count:.2%}")

    print("\nTotal Action Counts Across All Episodes:")
    total_actions = sum(total_action_counts.values())
    for action, count in sorted(total_action_counts.items()):
        print(f"  Action {action}: {count} ({(count / total_actions) * 100:.2f}%)")

    # Additional statistics
    print("\nAdditional Statistics:")
    print(f"  Return std: {np.std(all_episode_returns):.2f}")
    print(f"  Episode length std: {np.std(all_episode_lengths):.2f}")
    print(f"  Speed std: {np.std(episode_speeds):.2f}")

    # Find the episode numbers for min/max statistics
    min_return_episode_idx = np.argmin(all_episode_returns)
    max_return_episode_idx = np.argmax(all_episode_returns)
    min_length_episode_idx = np.argmin(all_episode_lengths)
    max_length_episode_idx = np.argmax(all_episode_lengths)

    print(
        f"  Min return: {all_episode_returns[min_return_episode_idx]:.2f} (episode {min_return_episode_idx + 1})"
    )
    print(
        f"  Max return: {all_episode_returns[max_return_episode_idx]:.2f} (episode {max_return_episode_idx + 1})"
    )
    print(
        f"  Min episode length: {all_episode_lengths[min_length_episode_idx]} (episode {min_length_episode_idx + 1})"
    )
    print(
        f"  Max episode length: {all_episode_lengths[max_length_episode_idx]} (episode {max_length_episode_idx + 1})"
    )

    if record_videos:
        print(f"\nRecorded videos saved to {video_folder}")

    return {
        "episode_count": episode_count,
        "total_collisions": total_collisions,
        "collision_rate": total_collisions / episode_count,
        "success_rate": (episode_count - total_collisions) / episode_count,
        "mean_return": np.mean(all_episode_returns),
        "mean_length": np.mean(all_episode_lengths),
        "mean_speed": np.mean(episode_speeds),
        "episode_returns": all_episode_returns,
        "episode_lengths": all_episode_lengths,
        "episode_speeds": episode_speeds,
        "action_counts": dict(total_action_counts),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Evaluate a trained SB3 DQN agent on the {parameters['env_name']} environment.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sb3/models/best_model.zip",
        help="The path to the SB3 model to load (default: sb3/models/best_model.zip).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100).",
    )
    parser.add_argument(
        "--random-agent",
        action="store_true",
        help="Evaluate using a random agent instead of a trained model.",
    )
    parser.add_argument(
        "-r",
        "--record",
        action="store_true",
        help="Record videos of the evaluation episodes.",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="sb3/videos/sb3_eval",
        help="Folder to save videos (default: videos/sb3_eval).",
    )
    parser.add_argument(
        "--save-stats",
        type=str,
        default="sb3_evaluation_results.txt",
        help="Save evaluation statistics to a file (default: sb3_evaluation_results.txt).",
    )

    args = parser.parse_args()

    try:
        # Run evaluation
        results = evaluate_sb3_model(
            model_path=args.model,
            num_episodes=args.episodes,
            record_videos=args.record,
            video_folder=args.video_folder,
            use_random_agent=args.random_agent,
        )

        # Save statistics to file if requested
        if args.save_stats:
            with open(args.save_stats, "w") as f:
                f.write("SB3 Model Evaluation Results\n")
                f.write(f"{'=' * 40}\n")
                if args.random_agent:
                    f.write("Using Random Agent\n")
                else:
                    f.write(f"Model: {args.model}\n")
                f.write(f"Episodes: {args.episodes}\n")
                f.write(f"Environment: {parameters['env_name']}\n\n")
                f.write("Summary Statistics:\n")
                f.write(f"  Episodes: {results['episode_count']}\n")
                f.write(f"  Collisions: {results['total_collisions']}\n")
                f.write(f"  Collision Rate: {results['collision_rate']:.2%}\n")
                f.write(f"  Success Rate: {results['success_rate']:.2%}\n")
                f.write(f"  Mean Return: {results['mean_return']:.2f}\n")
                f.write(f"  Mean Length: {results['mean_length']:.2f}\n")
                f.write(f"  Mean Speed: {results['mean_speed']:.2f}\n\n")
                f.write("Action Distribution:\n")
                total_actions = sum(results["action_counts"].values())
                for action, count in sorted(results["action_counts"].items()):
                    f.write(
                        f"  Action {action}: {count} ({(count / total_actions) * 100:.2f}%)\n"
                    )
            print(f"\nStatistics saved to {args.save_stats}")

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        exit(1)
