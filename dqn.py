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
    def __init__(self, env: gym.Env, mem_buffer: ExperienceReplay):
        """Initializes the agent.

        Args:
            `env`: The environment in which the agent will act.
            `mem_buffer`: The experience replay buffer.
        """
        self.env = env
        self.mem_buffer = mem_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, info = self.env.reset()
        self.total_reward = 0.0
        self.crashed = info.get("crashed")

    @torch.no_grad()
    def play_step(
        self, net: DQN, device: torch.device, epsilon: float = 0.0
    ) -> tt.Tuple[tt.Optional[float], bool]:
        """Execute one step of the agent in the environment.

        The agent selects and executes actions
        according to an epsilon-greedy policy based on Q.
        Args:
            `net`: The neural network that estimates the Q-values.
            `device`: The device on which the network is stored.
            `epsilon`: The probability of selecting a random action.

        Returns:
            A tuple containing the total reward and a boolean indicating
            whether the agent crashed during the episode. If the episode
            is not over, the total reward is `None`.
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

        if info.get("crashed"):
            self.crashed = True

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
        crashed = self.crashed
        if is_done or is_trunc:  # end of episode
            done_reward = self.total_reward
            self._reset()
        return done_reward, crashed, info


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
    assert total_time < 24 * 60 * 60, (
        f"Total training time exceeds 24 hours: {total_time}"
    )
    print(f"Total training time: {time.strftime('%Hh %Mm', time.gmtime(total_time))}")


if __name__ == "__main__":
    # Assuming that credentials are set in the environment
    run = neptune.init_run(
        tags=["Highway", "Thesis params"],
        dependencies="environment.yaml",
        # Replace the `monitoring/<hash>/` pattern to make comparison easier
        monitoring_namespace="monitoring",
        source_files=[
            "dqn.py",
            "hyperparameters.py",
            "wrappers.py",
            "experience_replay.py",
            "model.py",
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: '{device}'")

    env = make_env(
        parameters["env_name"],
        m=parameters["agent_history_length"],
        render_mode="rgb_array",  # For video recording
    )
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos/training",
        name_prefix="training",
        episode_trigger=lambda x: x % 90 == 0,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # 1. Initialize replay memory D to its capacity
    replay_memory = ExperienceReplay(parameters["replay_size"])

    expected_shape = (parameters["action_repeat"], 128, 64)
    assert obs_shape == expected_shape, f"Expected {expected_shape}, got {obs_shape}."

    # 2. Initialize action-value function Q with random weights
    policy_net = DQN(obs_shape, n_actions).to(device)
    # 3. Initialize target action-value function with weights of action-value function Q
    tgt_net = DQN(obs_shape, n_actions).to(device)
    tgt_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(
        policy_net.parameters(), lr=parameters["learning_rate"]
    )

    # Add Neptune logging
    npt_logger = NeptuneLogger(
        run=run,
        model=policy_net,
        log_model_diagram=True,
        log_gradients=True,
        log_parameters=True,
        # FIXME underneath is maybe too high?
        log_freq=500,  # Fixed the error with time stamps being too close
    )
    run[npt_logger.base_namespace]["hyperparameters"] = stringify_unsupported(
        parameters
    )

    agent = Agent(env, replay_memory)

    total_rewards = []
    crash_c = 0
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    timer = time.time()  # For measuring total training time
    best_mean_reward = None

    # 4. For each episode
    while frame_idx < parameters["max_frames"]:
        frame_idx += 1
        # 5. Initialize frame sequence and preprocessed sequence
        # 6. For each time step - done in `play_step` per `_reset`
        epsilon = calc_eps(frame_idx)
        reward, crashed, info = agent.play_step(policy_net, device, epsilon)
        if crashed:
            crash_c += 1
        if reward is not None:
            # Report progress at the end of the episode
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            # Mean reward of the last 100 episodes
            mean_reward = np.mean(total_rewards[-100:])
            epoch = len(total_rewards)
            crash_rate = crash_c / epoch
            print(
                f"{frame_idx}: done {epoch} games, "
                f"mean reward {mean_reward:.3f}, eps {epsilon:.2f}, "
                f"speed {speed:.2f} f/s,",
                f"crashes per episode {crash_rate:.2f}",
            )
            run[npt_logger.base_namespace]["metrics/epsilon"].append(
                epsilon, step=epoch
            )
            run[npt_logger.base_namespace]["metrics/speed"].append(speed, step=epoch)
            run[npt_logger.base_namespace]["metrics/reward_100"].append(
                mean_reward, step=epoch
            )
            run[npt_logger.base_namespace]["metrics/reward"].append(reward, step=epoch)
            run[npt_logger.base_namespace]["metrics/crash_rate"].append(
                crash_rate, step=epoch
            )
            run[npt_logger.base_namespace]["metrics/crashed"].append(
                float(crashed), step=epoch
            )
            run[npt_logger.base_namespace]["metrics/total_crashes"].append(
                crash_c, step=epoch
            )
            run[npt_logger.base_namespace]["metrics/episode_info"].append(
                info["episode"], step=epoch
            )
            if best_mean_reward is None or best_mean_reward < mean_reward:
                file_name = parameters["env_name"] + f"-best_{mean_reward:.0f}.dat"
                # Save model params
                torch.save(policy_net.state_dict(), file_name)
                if best_mean_reward is not None:
                    print(
                        f"Best mean reward updated "
                        f"{best_mean_reward:.3f} -> {mean_reward:.3f}"
                    )
                    best_mean_reward = mean_reward
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

    record_time(timer)
    run[npt_logger.base_namespace]["models"].upload_files("*.dat")
    npt_logger.log_model("latest_model")  # TODO also locally?
    run.stop()
    env.close()
