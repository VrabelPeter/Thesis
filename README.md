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
- [`hyperparameters.py`](hyperparameters.py) - Configuration parameters for experiment setup
- [`wrappers.py`](wrappers.py) - Wrappers for the `highway-fast-v0` environment
- [`train.py`](train.py) - Main training script for the DQN agent from SB3
- [`evaluate.py`](evaluate.py) - Script for evaluating a trained model

# Usage

## Training

Run training with default parameters:
```bash
python train.py
```

With custom seed:
```bash
python train.py --seed 123
```

With video recording:
```bash
python train.py --record videos/train
```

## Evaluation

Run evaluation with default parameters:
```bash
python evaluate.py
```

With video recording:
```bash
python evaluate.py --record --video-folder videos/eval
```

Evaluate a specific model with a different number of episodes and save the results:
```bash
python evaluate.py --model models/best_model.zip --episodes 500 --save-stats reports/evaluation.txt
```

Evaluate a random agent:
```bash
python evaluate.py --random-agent
```

# Citing
