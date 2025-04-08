

# Installation

The implementation was tested only with Python 3.11.

Using pip
```bash
pip install -r requirements.txt
```

Using Conda
```bash
conda env create -f environment.yaml
conda activate thesis
```

# Project structure
- [`environment.yaml`](environment.yaml) - Conda environment specification
- [`requirements.txt`](requirements.txt) - Python package dependencies
- [`model.py`](model.py) - DQN neural network architecture
- [`wrappers.py`](wrappers.py) - Wrappers for the `highway-v0` environment
- [`experience_replay.py`](experience_replay.py) - Experience replay buffer implementation
- [`train.py`](train.py) - Main training script for the DQN agent
- [`test.py`](test.py) - Script for evaluating a trained model
- [`hyperparameters.py`](hyperparameters.py) - Configuration parameters for different experiment setups

# Usage

# Citing
