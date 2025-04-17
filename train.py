import argparse
import random
import time
import typing as tt

import gymnasium as gym
import neptune
import numpy as np
import torch
import torch.nn as nn
from neptune.utils import stringify_unsupported
from neptune_pytorch import NeptuneLogger

from experience_replay import Experience, ExperienceReplay
from hyperparameters import parameters
from model import DQN
from wrappers import make_env

BatchTensors = tt.Tuple[
    torch.ByteTensor,  # current state
    torch.LongTensor,  # actions
    torch.Tensor,  # rewards
    torch.BoolTensor,  # done || trunc
    torch.ByteTensor,  # next state
]


def calc_eps(frame_idx):
    """Compute the epsilon value for the current frame.

    Anneals the epsilon value linearly
    from `epsilon_start` to `epsilon_final`
    over the first `epsilon_decay_last_frame` frames.

    Args:
        `frame_idx`: The current frame index.
    """
    return max(
        parameters["epsilon_final"],
        parameters["epsilon_start"]
        - frame_idx / parameters["epsilon_decay_last_frame"],
    )


class Agent:
    def __init__(self, env: gym.Env, mem_buffer: ExperienceReplay, initial_seed: int):
        """Initializes the agent.

        Args:
            `env`: The environment in which the agent will act.
            `mem_buffer`: The experience replay buffer.
            `initial_seed`: The seed for the environment.
        """
        self.env = env
        self.mem_buffer = mem_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset(initial_seed)

    def _reset(self, seed: int = None):
        """Reset the environment and agent state.
        Args:
            `seed`: The seed for the environment.
                    Should only be provided when the agent is created.
        """
        self.state, _ = self.env.reset(seed=seed)
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(
        self, net: DQN, device: torch.device, epsilon: float = 0.0
    ) -> tt.Tuple[tt.Optional[float], dict]:
        """Execute one step of the agent in the environment.

        The agent selects and executes actions
        according to an epsilon-greedy policy based on Q.
        Args:
            `net`: The neural network that estimates the Q-values.
            `device`: The device on which the network is stored.
            `epsilon`: The probability of selecting a random action.

        Returns:
            A tuple containing the total reward received if the episode is done,
            and the information dictionary from the environment.
        """
        if random.random() < epsilon:
            # 7. With probability epsilon select a random action
            action = self.env.action_space.sample()
        else:
            # 7. Otherwise select $a = argmax_a Q(\phi(s), a; \theta)$
            state_v = torch.as_tensor(self.state).to(device)
            # Add a batch dimension for the network
            state_v.unsqueeze_(0)
            # Q(\phi(s), a; \theta)
            q_vals_v = net(state_v)  # (1, n_actions)
            # a = argmax_a Q(\phi(s), a; \theta)
            act_v = torch.argmax(q_vals_v, dim=1)
            # Convert to scalar
            action = int(act_v.item())

        # 8. Execute action a in emulator
        # and observe reward r and image x'
        new_state, reward, is_done, is_trunc, info = self.env.step(action)
        self.total_reward += reward

        # 10. Store transition (phi, a, r, phi') in D
        exp = Experience(
            state=self.state,
            action=action,
            reward=float(reward),
            is_done=is_done or is_trunc,
            new_state=new_state,
        )
        self.mem_buffer.store(exp)

        self.state = new_state
        done_reward = None
        if is_done or is_trunc:  # end of episode
            done_reward = self.total_reward
            self._reset()
        return done_reward, info


def batch_to_tensors(
    batch: tt.List[Experience],
    device: torch.device,
) -> BatchTensors:
    """Convert a batch of experiences to corresponding tensors and
    move them to the device."""
    states, actions, rewards, dones, next_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.is_done)
        next_states.append(exp.new_state)
    # Try to avoid making a copy of the data by using `np.asarray`
    states_t = torch.as_tensor(np.asarray(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_t = torch.FloatTensor(rewards).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    next_states_t = torch.as_tensor(np.asarray(next_states)).to(device)
    return states_t, actions_t, rewards_t, dones_t, next_states_t


def calc_loss(
    batch: tt.List[Experience],
    policy_net: DQN,
    tgt_net: DQN,
    device: torch.device,
) -> torch.Tensor:
    """Calculate the loss for the batch of experiences.

    Args:
        `batch`: A batch of experiences.
        `policy_net`: The network that estimates the Q-values.
        `tgt_net`: The target network that estimates the Q-values.
        `device`: The device on which the networks are stored.

    Returns:
        The loss value on which to perform backpropagation.
    """
    states_t, actions_t, rewards_t, dones_t, next_states_t = batch_to_tensors(
        batch, device
    )
    # Model computes Q-values for all actions
    q_vals = policy_net(states_t)  # (batch_size, n_actions)
    # We select the Q-values for the actions taken
    q_vals = q_vals.gather(
        dim=1,
        index=actions_t.unsqueeze(-1),
    )  # (batch_size, 1)
    q_vals = q_vals.squeeze(-1)  # (batch_size,)
    with torch.no_grad():
        next_q_vals = tgt_net(next_states_t)  # (batch_size, n_actions)
        # max_a Q(s', a; \theta')
        next_q_vals = next_q_vals.max(dim=1)[0]  # (batch_size,)
        # 12. Set y = r for terminal s'
        next_q_vals[dones_t] = 0.0
        # Prevent gradients from flowing through the target network
        target_q_vals = next_q_vals.detach()
    # 12. Set y = r + \gamma max_a Q(s', a; \theta') for non-terminal s'
    expected_q_vals = rewards_t + parameters["gamma"] * target_q_vals

    criterion = nn.MSELoss()
    assert q_vals.shape == expected_q_vals.shape, (
        f"{q_vals.shape} != {expected_q_vals.shape}"
    )
    return criterion(q_vals, expected_q_vals)  # (y - Q(s, a; \theta))^2


def record_time(timer):
    """
    Record and display the total training time in hours and minutes.

    Parameters:
        timer (float): The start time in seconds (as returned by time.time())
            marking the beginning of the training period.

    Raises:
        AssertionError: If the total elapsed time (since timer) exceeds 24 hours.

    Prints:
        A formatted string showing the total training duration in the format 'HHh MMm'.
    """
    total_time = int(time.time() - timer)
    print(f"Total training time: {time.strftime('%Hh %Mm', time.gmtime(total_time))}")


def log_metrics(
    run: neptune.Run,
    npt_logger: NeptuneLogger,
    env: gym.Env,
    frame_idx: int,
    episode_c: int,
    crash_c: int,
    epsilon: float,
    speed_values: tt.List[float],
    info: dict,
):
    """Log episode metrics to the console and Neptune.

    Args:
        run: The Neptune run object for logging.
        npt_logger: The Neptune logger object.
        env: The environment instance.
        frame_idx: The current frame index.
        episode_c: The current episode count.
        crash_c: The current crash count.
        epsilon: The current epsilon value.
        speed_values: A list of speed values for the current episode.
        info: The `info` dictionary from the environment step.
    """
    episode_info = info["episode"]
    episode_return = episode_info["r"]  # Cumulative reward
    episode_length = episode_info["l"]  # Episode length in steps
    # Time elapsed since beginning of the episode
    episode_time = episode_info["t"]
    episode_mean_speed = np.mean(speed_values) if speed_values else 0.0
    # Means of the last 100 episodes
    mean_return_100 = np.mean(env.return_queue)
    mean_length_100 = np.mean(env.length_queue)
    mean_time_100 = np.mean(env.time_queue)
    assert episode_c > 0, "Episode count must be greater than 0."
    crash_rate = crash_c / episode_c
    assert episode_time > 0, "Episode time must be greater than 0."
    speed = episode_length / episode_time

    print(
        f"{frame_idx}: done {episode_c} games,",
        f"mean reward {mean_return_100:.3f},",
        f"eps {epsilon:.2f},",
        f"speed {speed:.2f} f/s,",
        f"crashes per episode {crash_rate:.2f}",
    )

    run[npt_logger.base_namespace]["metrics"].append(
        {
            "epsilon": epsilon,
            "speed": speed,
            "total_crashes": crash_c,
            "crash_rate": crash_rate,
        }
    )
    run[npt_logger.base_namespace]["metrics/means_100"].append(
        {
            "return": mean_return_100,
            "length": mean_length_100,
            "time": mean_time_100,
        },
        step=episode_c,
    )
    run[npt_logger.base_namespace]["metrics/episode"].append(
        {
            "return": episode_return,
            "length": episode_length,
            "time": episode_time,
            "crashed": info["crashed"],
            "mean_speed": episode_mean_speed,
        },
        step=episode_c,
    )


def set_seed(seed: int) -> None:
    """Set the seed for reproducibility.
    Args:
        seed: The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train DQN agent")
    parser.add_argument(
        "-r",
        "--record",
        type=str,
        help="Record video of the training to the specified folder",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed to use for reproducibility, default=42",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    # Assuming that credentials are set in the environment
    run = neptune.init_run(
        tags=["Highway", "Thesis params"],
        dependencies="environment.yaml",
        # Replace the `monitoring/<hash>/` pattern to make comparison easier
        monitoring_namespace="monitoring",
        source_files=[
            "train.py",
            "hyperparameters.py",
            "wrappers.py",
            "experience_replay.py",
            "model.py",
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: '{device}'")
    # For video recording, otherwise None
    render_mode = "rgb_array" if args.record else None
    env = make_env(
        parameters["env_name"],
        m=parameters["agent_history_length"],
        video_folder=args.record,
        name_prefix="train",
        record_frequency=100,
        buffer_length=100,
        render_mode=render_mode,
    )
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    # 1. Initialize replay memory D to its capacity
    replay_memory = ExperienceReplay(parameters["replay_size"])
    expected_shape = (parameters["agent_history_length"], 128, 64)
    assert obs_shape == expected_shape, f"Expected {expected_shape}, got {obs_shape}."
    # 2. Initialize action-value function Q with random weights
    policy_net = DQN(obs_shape, n_actions).to(device)
    # 3. Initialize target action-value function with weights of action-value function Q
    tgt_net = DQN(obs_shape, n_actions).to(device)
    tgt_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(
        policy_net.parameters(), lr=parameters["learning_rate"]
    )
    npt_logger = NeptuneLogger(run=run, model=policy_net)
    run[npt_logger.base_namespace]["hyperparameters"] = stringify_unsupported(
        parameters
    )
    run[npt_logger.base_namespace]["seed"] = args.seed
    agent = Agent(env, replay_memory, args.seed)
    episode_c = 0
    crash_c = 0
    frame_idx = 0
    total_training_timer = time.time()
    best_mean_reward = None
    speed_values = []
    # 4. For each episode
    while frame_idx < parameters["max_frames"]:
        frame_idx += 1
        # 5. Initialize frame sequence and preprocessed sequence
        # 6. For each time step - done in `play_step` per `_reset`
        epsilon = calc_eps(frame_idx)
        reward, info = agent.play_step(policy_net, device, epsilon)
        if info["crashed"]:
            crash_c += 1  # Unsuccessful episode
        speed_values.append(info["speed"])
        if reward is not None:
            episode_c += 1  # End of episode
            # Report progress
            log_metrics(
                run,
                npt_logger,
                env,
                frame_idx,
                episode_c,
                crash_c,
                epsilon,
                speed_values,
                info,
            )
            mean_return_100 = np.mean(env.return_queue)
            if best_mean_reward is None or best_mean_reward < mean_return_100:
                file_name = f"{parameters['env_name']}-best_{mean_return_100:.0f}.dat"
                # Save model params
                torch.save(policy_net.state_dict(), file_name)
                if best_mean_reward is not None:
                    print(
                        f"Best mean reward updated "
                        f"{best_mean_reward:.3f} -> {mean_return_100:.3f}"
                    )
                    best_mean_reward = mean_return_100
            speed_values = []
        if len(replay_memory) < parameters["replay_start_size"]:
            continue
        # 14. Every C steps reset \theta' = \theta.
        # Due to increment placement sync frames before loss calculation
        if frame_idx % parameters["sync_target_frames"] == 0:
            tgt_net.load_state_dict(policy_net.state_dict())
        optimizer.zero_grad()
        # 11. Sample random mini-batch of transitions from D
        batch = replay_memory.sample(parameters["batch_size"])
        loss_t = calc_loss(batch, policy_net, tgt_net, device)
        # 13. Perform a GD step on (y - Q(s, a; \theta))^2
        loss_t.backward()
        optimizer.step()
    # End of training
    record_time(total_training_timer)
    torch.save(
        policy_net.state_dict(),
        f"{parameters['env_name']}-final_seed_{args.seed}.dat",
    )
    run[npt_logger.base_namespace]["models"].upload_files("*.dat")
    run.stop()
    env.close()
