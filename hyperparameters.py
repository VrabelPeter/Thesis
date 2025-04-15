base_parameters = {
    "env_name": "highway-v0",
    "activation": "ReLU",
}

publication_parameters = base_parameters.copy()
publication_parameters.update(
    {
        # Requires ~58 MB of VRAM (see `mem_req_resarch.ipynb`)
        "batch_size": 32,
        # Requires ~56 GB of RAM (as one is ~56 KB, see `experience_replay.py`)
        "replay_size": 1_000_000,
        "agent_history_length": 4,
        "sync_target_frames": 10_000,
        "gamma": 0.99,  # discount factor
        "action_repeat": 4,
        "learning_rate": 0.000_25,  # Uses RMSprop
        "gradient_momentum": 0.95,
        "squared_gradient_momentum": 0.95,
        "min_squared_gradient": 0.01,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "epsilon_decay_last_frame": 1_000_000,
        "replay_start_size": 50_000,
        "noop_max": 30,
        "max_frames": 50_000_000,
    }
)

thesis_params = publication_parameters.copy()
thesis_params["replay_size"] = 100_000
thesis_params["learning_rate"] = 3e-3  # Uses Adam
thesis_params["gamma"] = 0.99
thesis_params["sync_target_frames"] = 250
thesis_params["epsilon_decay_last_frame"] = 24_000  # Should be 12% of max_frames
thesis_params["epsilon_final"] = 0.04
thesis_params["replay_start_size"] = 1_000
thesis_params["batch_size"] = 64
thesis_params["max_frames"] = 200_000

# Change only this line to switch between parameters
parameters = thesis_params
