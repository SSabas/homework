#!/bin/bash
set -eux
for e in Hopper Ant HalfCheetah Humanoid Reacher Walker2d
do
    python run_bc.py $e-v2 --batch_size 256 --epochs 100 --val_prop 0.1 --layers 100 100 100 --act_function 'relu' --verbose 'yes' --plot 'yes' --save_model 'yes' --save_results 'yes'
done


