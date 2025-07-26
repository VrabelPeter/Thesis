import argparse
import os
import signal
from collections import deque

import gymnasium as gym
import neptune
import numpy as np
import torch
from neptune.utils import stringify_unsupported
from neptune_tensorboard import enable_tensorboard_logging
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback

from hyperparameters import parameters
from wrappers import create_sb3_env, make_base_env, wrap_env_with_recorder

TS_LOG_DIR = "ts_logs/"
MODELS_DIR = "models/"

interrupted = False


def signal_handler(signum, frame):
    global interrupted
    print("\nReceived Ctrl+C. Gracefully stopping training...")
    interrupted = True


# Set up the signal handler
signal.signal(signal.SIGINT, signal_handler)


class CustomCallback(BaseCallback):
    """
    Custom callback for logging and monitoring training progress.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.mean_100_speeds = deque(maxlen=100)
        self.crashes_count = 0
        self.ep_c = 0  # Episode counter

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        global interrupted
        if interrupted:
            print("Training interrupted by user.")
            return False

        info = self.locals.get("infos", [])[-1] if self.locals.get("infos") else {}
        if "episode" in info:  # End of episode marked by the `Monitor` wrapper
            self.ep_c += 1
            ep_metrics = info.get("episode_metrics", {})
            if ep_metrics:
                self.mean_100_speeds.append(ep_metrics.get("mean_speed", 0))
                self.crashes_count += int(ep_metrics.get("crashed", False))
                if self.verbose >= 1:
                    print(
                        f"Episode finished: mean speed = {np.mean(self.mean_100_speeds):.2f}, "
                        f"crashed = {ep_metrics.get('crashed', False)}"
                    )

                if self.ep_c % 100 == 0:
                    self.logger.record("training/total_crashes", self.crashes_count)
                    self.logger.record(
                        "training/crash_rate",
                        self.crashes_count / self.ep_c,
                    )
                    if self.mean_100_speeds:
                        mean_speed = np.mean(self.mean_100_speeds)
                        self.logger.record("training/mean_100_speed", mean_speed)
        return True


class VideoRecordingCallback(BaseCallback):
    """
    Callback for saving a video of the agent.
    It is triggered by the `EvalCallback` when a new best model is found.

    :param video_folder: Folder to save videos
    """

    def __init__(self, video_folder: str, n_episodes: int = 10, verbose=0):
        super().__init__(verbose)
        self.video_folder = video_folder
        self.n_episodes = n_episodes

    def _on_step(self) -> bool:
        global interrupted
        if interrupted:
            print("Training interrupted by user.")
            return False

        eval_env = make_base_env(
            env_name=parameters["env_name"],
            m=parameters["agent_history_length"],
            render_mode="rgb_array",
        )
        eval_env = wrap_env_with_recorder(
            eval_env,
            name_prefix=f"best_model_{self.num_timesteps}",
            video_folder=self.video_folder,
            record_frequency=1,
        )
        eval_env = gym.wrappers.RecordEpisodeStatistics(
            eval_env, buffer_length=self.n_episodes
        )

        mean_speeds = []
        total_crashes = 0
        for _ in range(self.n_episodes):
            obs, info = eval_env.reset()
            done = False
            ep_speeds = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, is_trunc, info = eval_env.step(action)
                done = done or is_trunc

                ep_speeds.append(info.get("speed", 0))

            crashed = info.get("crashed", False)
            if crashed:
                total_crashes += 1
            mean_speed = np.mean(ep_speeds) if ep_speeds else 0
            mean_speeds.append(mean_speed)

        mean_reward = np.mean(eval_env.return_queue)
        mean_length = np.mean(eval_env.length_queue)
        success_rate = 1 - (total_crashes / self.n_episodes)
        mean_speed = np.mean(mean_speeds) if mean_speeds else 0

        self.logger.record("best_model_eval/mean_reward", mean_reward)
        self.logger.record("best_model_eval/mean_length", mean_length)
        self.logger.record("best_model_eval/success_rate", success_rate)
        self.logger.record("best_model_eval/mean_speed", mean_speed)

        eval_env.close()

        if self.verbose > 0:
            print(f"Videos saved to {self.video_folder}")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DQN agent using Stable Baselines 3"
    )
    parser.add_argument(
        "-r",
        "--record",
        default="videos/best_model",
        type=str,
        help="Record videos of best models to the specified folder",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed to use for reproducibility, default=42",
    )
    args = parser.parse_args()
    # Assuming that credentials are set in the environment
    run = neptune.init_run(
        tags=[
            parameters["env_name"],
            "Publication params",
            "SB3",
        ],
        dependencies="requirements.txt",
        # Replace the `monitoring/<hash>/` pattern to make comparison easier
        monitoring_namespace="monitoring",
        source_files=[
            "hyperparameters.py",
            "wrappers.py",
            "train.py",
        ],
    )
    enable_tensorboard_logging(run)
    run["training/hyperparameters"] = stringify_unsupported(parameters)
    run["training/seed"] = args.seed

    os.makedirs(TS_LOG_DIR, exist_ok=True)  # Create log directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run["training/device"] = device

    env = create_sb3_env(
        env_name=parameters["env_name"],
        m=parameters["agent_history_length"],
    )

    eval_env = create_sb3_env(
        env_name=parameters["env_name"],
        m=parameters["agent_history_length"],
    )

    # Instantiate the agent
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=parameters["learning_rate"],
        buffer_size=parameters["replay_size"],
        learning_starts=parameters["replay_start_size"],
        batch_size=parameters["batch_size"],
        gamma=parameters["gamma"],
        train_freq=1,  # Every fourth step is not applicable for this domain
        replay_buffer_kwargs=dict(
            handle_timeout_termination=False  # For optimized replay buffer
        ),
        optimize_memory_usage=True,  # Allows me a replay buffer size of 1M elements
        target_update_interval=parameters["sync_target_frames"],
        exploration_fraction=parameters["epsilon_decay_last_frame"]
        / parameters["training_frames"],
        exploration_initial_eps=parameters["epsilon_start"],
        exploration_final_eps=parameters["epsilon_final"],
        tensorboard_log=TS_LOG_DIR,  # By default statistics are computed over 100 episodes
        policy_kwargs=dict(
            optimizer_class=torch.optim.RMSprop,
            optimizer_kwargs=dict(
                momentum=parameters["gradient_momentum"],
                alpha=parameters["squared_gradient_momentum"],
                eps=parameters["min_squared_gradient"],
            ),
        ),
        # verbose=1, # Prints training information onto stdout
        verbose=0,  # No output
        seed=args.seed,  # Sets all the seeds for reproducibility
        device=device,
    )

    print(f"Optimizer used: {model.policy.optimizer.__class__.__name__}")

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=VideoRecordingCallback(args.record),
        n_eval_episodes=100,  # Want optimal behavior for at least 80%
        eval_freq=250_000,  # Totals 120 evaluations
        log_path=TS_LOG_DIR,
        best_model_save_path=f"{MODELS_DIR}/{args.seed}",
    )
    custom_callback = CustomCallback()
    callback_list = CallbackList([eval_callback, custom_callback])

    try:
        # Train the agent and display a progress bar
        model.learn(
            total_timesteps=parameters["training_frames"],
            callback=callback_list,
            log_interval=100,
            progress_bar=True,
        )

    except KeyboardInterrupt:
        print("Training interrupted by user")

    # Save final model
    final_model_path = f"models/{parameters['env_name']}-final_seed_{args.seed}"
    model.save(final_model_path)  # Save the agent
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved in {MODELS_DIR}")

    # Upload model to Neptune
    run["models/final"].upload(f"{final_model_path}.zip")

    run.stop()
    env.close()

    print("Training completed!")
    print(f"To view logs, run: tensorboard --logdir {TS_LOG_DIR}")
