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

## Using This Repo

I have been using conda to manage dependencies.

```bash
conda env create -f environment.yml
conda activate rl-infra
git clone https://github.com/jonathanlamar/tetris.git tetris
pip install tetris/
pip install .
```

If you are going to play with the tetris implementation, you will need to clone
[the tetris repo](https://github.com/jonathanlamar/tetris) and run `pip install .` from the root of it in your conda
environment.
