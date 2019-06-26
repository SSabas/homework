#!/bin/bash
set -eux
for e in Hopper Ant HalfCheetah Humanoid Reacher Walker2d
do
    python run_eval_bc.py $e-v2
done
