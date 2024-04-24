# RL Infrastructure

This is a repo containing infra code for my ongoing RL side project. The purpose of this project is to demonstrate how
a service architecture might be designed for maintaining a full stack reinforcement learning product at scale, although
the example implementations of the design do not necessarily run at scale. The idea here is that the engineering
challenge behind machine learning is not in the code for running or training a model, but rather the services for
collecting and preparing data for the model, versioning and serving trained models to the edge, and tracking online
performance of the models in production.

See [this blog post](https://www.jonalarm.com/https://www.jonalarm.com/infra/infra_1/) for more information about this
project.

## Abstraction

One of the main goals was to write a system of interfaces that could support multiple use-cases. The structure of the
repo therefore reflects this decision. In
[rl_infra/types](https://github.com/jonathanlamar/rl-infra/tree/main/rl_infra/types), you can find all of the interfaces
that need toi be implemented for a new use-case. In
[rl_infra/impl](https://github.com/jonathanlamar/rl-infra/tree/main/rl_infra/impl), you can find the existing
implementations that I used. So far, the tetris implementation is the most complete, but there has been partial work on
a GoPiGo implementation that I hope to complete after tetris. I would also like to use an OpenAI Gym environment in a
third implementation.

## Using This Repo to Train a Tetris Bot

I have been using conda to manage dependencies, so you will need that installed on your PATH.  Assuming that is the
case, simply run `./bin/install.sh` to create the environment and build the package into that environment. To clean up
build artifacts, run `./bin/cleanup.sh`.

To initialize a model, use `./bin/cold_start_tetris.py`.  This script accepts a model tag parameter if you want to
version multiple architectures simultaneously. To train a model, use `./bin/train_tetris.py`.  Use the help strings for
both of these scripts to understand how to use them.  It goes without saying, but you need to activate the rl-infra
conda environment prior to running either of them.
