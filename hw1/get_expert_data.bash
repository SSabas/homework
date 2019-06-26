#!/bin/bash
set -eux
mkdir -p expert_data
for e in Hopper Ant HalfCheetah Humanoid Reacher Walker2d
do
    python run_expert.py experts/$e-v2.pkl $e-v2 --num_rollouts=20
done
