#!/usr/bin/env bash

rm -rf build tetris rl_infra.egg-info
find . -name __pycache__ | xargs rm -rf
