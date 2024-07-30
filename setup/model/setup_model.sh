#!/bin/bash

# setup.sh - set up conda environment and install dependencies

ENV_NAME="bcresnet_model"
PYTHON_VERSION="3.8.19"

# check if conda is installed first
check_conda_installed() {
    if ! command -v conda &> /dev/null; then
        echo "Conda is not installed. Please install Miniconda or Anaconda first."
        exit 1
    fi
}

check_conda_installed

# create conda environment if it doesn't exist
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

# activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# conda packages
conda install -y -c pytorch pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
conda install -y --file conda-requirements.txt

# pip packages
pip install -r requirements.txt

# system dependencies
sudo apt-get install -y libportaudio2

echo "Setup complete. The conda environment '$ENV_NAME' is ready with Python $PYTHON_VERSION."
