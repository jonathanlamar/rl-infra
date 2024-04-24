#!/usr/bin/env bash

conda env create -f environment.yml

# This is stupid, but I don't know a better way
CONDA_ENV_LOCATION=$(conda env list | grep "rl-infra" | rev | cut -d " " -f 1 | rev)

git clone https://github.com/jonathanlamar/tetris.git tetris
$CONDA_ENV_LOCATION/bin/pip install tetris/
$CONDA_ENV_LOCATION/bin/pip install .
