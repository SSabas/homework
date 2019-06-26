# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.

**Note**: MuJoCo requires activation key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.


### Getting expert data
To generate expert data for training run:

```bash
bash get_expert_data.bash
```
The script uses `run_expert.py` to generate 20 roll-outs (each with 1000 iteratiors, totaling 20K examples) of experts policies of aforementioned tasks. The data is saved to `experts_data/` in pickle files.

## Section 2 - Behavioral Cloning
### 2.1 Implementing behavioural cloning
To train the supervised model for behavioral cloning run:

```bash
bash fit_bc_model.bash
```
The script uses `run_bc.py` to train a set of neural network based on the gathered expert data. The standard training setting uses 64-64-64 layer structure with 64 batch size and total 100 epochs for training. The resulting models are saved to `models/` as h5 files and corresponding fitting results with graphs in `results/bc/`.

### 2.2 Testing behavioural cloning
To test the performance of the BC models run:

```bash
bash fit_bc_model.bash
```
The script uses `run_bc_eval.py` to evaluate the performance of behavioural cloning policies for all the specified environments. The results are saved to `results/bc_evaluation/` as pickle files.

To plot the results in table, one can tabulating function `plot_bc_results` from `run_bc_eval.py` (requires pytablewriter package):

|              |Mean(BC)|Std(BC)|Mean(Exp)|Std(Exp)|
|--------------|-------:|------:|--------:|-------:|
|Ant-v2        |  816.75| 47.528| 4828.226| 113.865|
|HalfCheetah-v2|  570.31|445.816| 4114.136|  85.164|
|Hopper-v2     |   59.00|  2.399| 3778.434|   3.326|
|Humanoid-v2   |  263.09| 10.617|10415.470|  62.776|
|Reacher-v2    |  -13.81|  4.424|   -4.397|   1.809|
|Walker2d-v2   |  106.29| 62.579| 5532.184|  57.344|

### 2.3 Analysis of hyperparamters
To test the performance of the BC models run:
