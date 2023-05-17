# Barlow-twins-BYOL

This is a student project that aims to implement neural networks using self-supervised learning on audio data. We intend to generate general-purpose representations for downstream tasks. Out inspiration for the project is the [BYOL-A project](https://github.com/nttcslab/byol-a).

## Requirements

The code has been tested with Python 3.10.0 on Ubuntu 22.04.1-LTS.
It should work on other platforms as well, but this has not been extensively tested.

The DVC pipeline is written and tested for Linux using a bash shell. As such, we suggest
using Git Bash when using Windows.

Pre-requisites:
- Install Python 3.10.

## Installation

Install the project by following these instructions:

```python
# Adapt the instructions as relevant to your environment.

# If using Ubuntu, you might need to install venv and pip through the package manager:
# sudo apt-get install python3-venv
# sudo apt-get install python3-pip

# Create a virtual environment in the project root.
python3 -m venv venv

# Activate the virtual environment.
source venv/bin/activate
# source venv/Scripts/activate  # Use this for Windows.

# Install a Python dependencies.
python3 install -e .
# pip install -e .[dev]  # Use this instead for development.
```

## Usage

The project is set up with DVC. As such, you can reproduce everything by running `dvc repro` after following the installation instructions. This will:
- Download and preprocess datasets:
  - Audioset Train
  - GTZAN
- Train a Barlow twins model on Audioset
- Run a linear evaluation on GTZAN
- Report test results
- 
### Training

Work in progress

### Evaluation

Work in progress

### Tensorboard logs

Work in progress

### Optuna dashboard

Work in progress