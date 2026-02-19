# DeepReach: A Deep Learning Approach to High-Dimensional Reachability
### [Project Page](http://people.eecs.berkeley.edu/~somil/index.html) | [Paper](https://arxiv.org/pdf/2011.02082.pdf)<br>

Repository Maintainers<br>
[Albert Lin](https://www.linkedin.com/in/albertkuilin/),
[Zeyuan Feng](https://thezeyuanfeng.github.io/),
[Javier Borquez](https://javierborquez.github.io/),
[Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html)<br>
University of Southern California

Original Authors<br>
[Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html),
Claire Tomlin<br>
University of California, Berkeley

(Still to come...) The Safe and Intelligent Autonomy (SIA) Lab at the University of Southern California
is still working on an easy-to-use DeepReach Python package which will follow much of the same organizational principles as
the [hj_reachability package in JAX](https://github.com/StanfordASL/hj_reachability) from the Autonomous Systems Lab at Stanford.
The future version will include the newest tips and tricks of DeepReach developed by SIA.

(In the meantime...) This branch provides a moderately refactored version of DeepReach to facilitate easier outside research on DeepReach.

## What's New: Temporal Consistency Training

This repository has been extended with **temporal consistency training**, a novel approach for learning value functions from trajectory data without requiring explicit dynamics models. This extension enables training DeepReach models using observed state transitions from trajectories, making it particularly useful for systems where dynamics are unknown or difficult to model analytically.

### Key Features

- **Temporal Consistency Loss**: Uses observed flow from trajectory data to enforce the Hamilton-Jacobi PDE residual `|dV/dt + ∇V·ẋ_obs|²` instead of requiring explicit dynamics
- **Flexible Boundary Conditions**: Configurable `t=tMin` boundary handling via `--tc_t0_mode` (weighted, fixed, or off)
- **RAM-Optimized Data Loading**: Efficient trajectory loading with `--load_trajectories_in_ram` for faster training
- **Training Split Support**: Use predefined train/test splits with `--use_shuffled_indices_only`
- **Enhanced Evaluation**: Comprehensive ROA (Region of Attraction) evaluation with calibration and threshold optimization

## Full Technical Notebook
For the complete implementation walkthrough (data path, model, losses, training flow, full dynamics math derivations, configs, and diagnostics), read:

- `MEGA_NOTEBOOK.md`

## High-Level Structure
The code is organized as follows:
* `dynamics/dynamics.py` defines the dynamics of the system.
* `experiments/experiments.py` contains generic training routines.
* `utils/modules.py` contains neural network layers and modules.
* `utils/dataio.py` loads training and testing data (including temporal consistency sampling).
* `utils/diff_operators.py` contains implementations of differential operators.
* `utils/losses.py` contains loss functions for the different reachability cases (including temporal consistency).
* `run_experiment.py` starts a standard DeepReach experiment run.
* `evaluation/eval_roa.py` evaluates learned value functions for ROA prediction.
* `scripts/inspect_pretrain_curriculum.py` inspects pretraining curriculum progress.

## External Tutorial
Follow along these [tutorial slides](https://docs.google.com/presentation/d/19zxhvZAHgVYDCRpCej2svCw21iRvcxQ0/edit?usp=drive_link&ouid=113852163991034806329&rtpof=true&sd=true) to get started, or continue reading below.

## Environment Setup
Create and activate a virtual python environment (env) to manage dependencies:
```bash
python -m venv env
source env/bin/activate  # On Linux/Mac
# or
env\Scripts\activate  # On Windows
```
Install DeepReach dependencies:
```bash
pip install -r requirements.txt
```
Install the appropriate PyTorch package for your system. For example, for a Windows system with CUDA 12.1:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Running a DeepReach Experiment

### Standard HJ PDE Training

`run_experiment.py` implements a standard DeepReach experiment. For example, to learn the value function for the avoid Dubins3D system with parameters `goalR=0.25`, `velocity=0.6`, `omega_max=1.1`, run:
```bash
python run_experiment.py --mode train --experiment_class DeepReach --dynamics_class Dubins3D --experiment_name dubins3d_tutorial_run --minWith target --goalR 0.25 --velocity 0.6 --omega_max 1.1 --angle_alpha_factor 1.2 --set_mode avoid
```
Note that the script provides many common training arguments, like `num_epochs` and the option to `pretrain`. For detailed training internals and option behavior, see `MEGA_NOTEBOOK.md`. `use_CSL` is an experimental training option (similar in spirit to actor-critic methods) being developed by SIA for improved value function learning.

## CartPole Hybrid Training

This repo includes an extended CartPole path with support for both traditional HJ PDE training and temporal consistency training.

### CartPole Dataset Structure

When `--dynamics_class CartPole` is selected, the dataset path is chosen automatically in `run_experiment.py`.

Expected dataset root (`--data_root`):

* `trajectories/sequence_*.txt` - Trajectory files containing state sequences
* Optional `roa_labels.txt` or `cal_set.txt` - Supervised labels for ROA classification (9-column format: x, θ, ẋ, θ̇, x_next, θ_next, ẋ_next, θ̇_next, label)
* `dataset_description.json` - Used to auto-load `gravity`, `cart_mass`, `pole_mass`, `pole_length` if not provided
* `train_test_splits/shuffled_indices.txt` - Optional train/test split indices

### Training Modes

#### 1. HJ PDE + Supervised Labels (Traditional)

This mode combines PDE residual loss with supervised ROA labels:

* `dynamics_class=CartPole` uses a CartPole dynamics model and Hamiltonian in `dynamics/dynamics.py`.
* Training data is loaded from files via `CartPoleDataset` in `utils/dataio.py`.
* If a supervised label file is present and `num_supervised > 0`, an additional supervised value MSE term is added during training (`roa_labels.txt` by default, or override with `--supervised_labels_file`).
* To reduce class-imbalance collapse, you can enable `--supervised_balanced_sampling` and/or set class weights with `--supervised_safe_weight` and `--supervised_unsafe_weight`.
* To make PDE state sampling approximately uniform over stored trajectory points, enable `--trajectory_uniform_sampling`.
* To run on a smaller trajectory subset, set `--max_trajectory_files` (e.g., `100`).

**Example: Small run with supervised labels**
```bash
python run_experiment.py --mode train \
  --experiment_class DeepReach \
  --dynamics_class CartPole \
  --experiment_name cartpole_small \
  --minWith target \
  --u_max 2000 \
  --x_bound 6 --xdot_bound 5 --thetadot_bound 5 \
  --set_mode avoid \
  --data_root /path/to/cartpole_pybullet \
  --supervised_labels_file cal_set.txt \
  --num_supervised 256 \
  --supervised_weight 1.0 \
  --supervised_balanced_sampling \
  --supervised_safe_weight 2.0 \
  --supervised_unsafe_weight 1.0 \
  --trajectory_uniform_sampling \
  --max_trajectory_files 100 \
  --tMin 0.0 --tMax 2.0 \
  --numpoints 2000 \
  --num_epochs 200 \
  --pretrain --pretrain_iters 200 \
  --lr 1e-4 \
  --num_hl 2 --num_nl 128 \
  --model sine
```

**Example: Large-scale training with curriculum (50k pretrain + 70k finetune)**
```bash
# Using config file
python run_experiment.py -c configs/cartpole_50k_70k.cfg --mode train

# Or using command line
python run_experiment.py --mode train \
  --experiment_class DeepReach \
  --dynamics_class CartPole \
  --experiment_name cartpole_50k_70k_trainsplit_ram \
  --minWith target \
  --u_max 2000 --x_bound 6 --xdot_bound 5 --thetadot_bound 5 \
  --set_mode avoid \
  --data_root /path/to/cartpole_pybullet \
  --trajectory_uniform_sampling \
  --use_shuffled_indices_only \
  --shuffled_indices_file train_test_splits/shuffled_indices.txt \
  --load_trajectories_in_ram \
  --num_supervised 128 \
  --supervised_labels_file cal_set.txt \
  --supervised_weight 1.0 \
  --tMin 0.0 --tMax 6.13 \
  --counter_start 0 --counter_end 70000 \
  --numpoints 1000 \
  --num_epochs 120000 \
  --pretrain --pretrain_iters 50000 \
  --lr 7e-5 \
  --num_hl 2 --num_nl 96 \
  --model sine \
  --device cuda:0 \
  --seed 0
```

#### 2. Temporal Consistency Training (New!)

Temporal consistency mode learns value functions directly from trajectory data without requiring explicit dynamics. This is particularly useful when:
- System dynamics are unknown or difficult to model
- You have abundant trajectory data from simulations or real systems
- You want to learn from observed state transitions

**Key Parameters:**
- `--training_objective temporal_consistency` - Enables temporal consistency mode
- `--tc_loss_weight` (default: 1.0) - Weight for the temporal PDE residual loss
- `--tc_anchor_weight` (default: 0.1) - Weight for `t=tMin` boundary loss when using `weighted` mode
- `--tc_t0_mode {weighted,fixed,off}` - Boundary condition handling:
  - `weighted`: Apply boundary loss at `t=tMin` with `tc_anchor_weight` scaling
  - `fixed`: Apply boundary loss at `t=tMin` with weight 1.0
  - `off`: No boundary loss at `t=tMin`

**How it works:**
- Samples state-time pairs `(x, t)` from trajectories
- Computes observed flow `ẋ_obs` from one-step trajectory differences
- Enforces temporal PDE: `dV/dt + min(0, ∇V·ẋ_obs) = 0`
- Loss: `|dV/dt + ∇V·ẋ_obs|²` (backup condition)

**Example: Temporal consistency with weighted t0 boundary**
```bash
python run_experiment.py --mode train \
  --experiment_class DeepReach \
  --dynamics_class CartPole \
  --experiment_name cartpole_temporal_weighted \
  --minWith target \
  --u_max 2000 \
  --x_bound 6 --xdot_bound 5 --thetadot_bound 5 \
  --set_mode avoid \
  --data_root /path/to/cartpole_pybullet \
  --dataset_class CartPoleDataset \
  --training_objective temporal_consistency \
  --deepreach_model exact \
  --tc_loss_weight 1.0 \
  --tc_t0_mode weighted \
  --tc_anchor_weight 0.1 \
  --trajectory_uniform_sampling \
  --load_trajectories_in_ram \
  --tMin 0.0 --tMax 6.13 \
  --numpoints 2000 \
  --num_epochs 40000 \
  --lr 7e-5 \
  --num_hl 2 --num_nl 96 \
  --model sine \
  --steps_til_summary 2000 \
  --epochs_til_ckpt 5000 \
  --device cuda:0 \
  --seed 0
```

**Example: Temporal consistency with fixed t0 boundary**
```bash
python run_experiment.py --mode train \
  --experiment_class DeepReach \
  --dynamics_class CartPole \
  --experiment_name cartpole_temporal_fixed \
  --minWith target \
  --u_max 2000 \
  --x_bound 6 --xdot_bound 5 --thetadot_bound 5 \
  --set_mode avoid \
  --data_root /path/to/cartpole_pybullet \
  --dataset_class CartPoleDataset \
  --training_objective temporal_consistency \
  --deepreach_model exact \
  --tc_loss_weight 1.0 \
  --tc_t0_mode fixed \
  --tc_anchor_weight 0.1 \
  --trajectory_uniform_sampling \
  --load_trajectories_in_ram \
  --tMin 0.0 --tMax 6.13 \
  --numpoints 2000 \
  --num_epochs 40000 \
  --lr 7e-5 \
  --num_hl 2 --num_nl 96 \
  --model sine \
  --device cuda:0 \
  --seed 0
```

**Example: Temporal consistency with supervised labels**
```bash
python run_experiment.py --mode train \
  --experiment_class DeepReach \
  --dynamics_class CartPole \
  --experiment_name cartpole_temporal_with_labels \
  --minWith target \
  --u_max 2000 \
  --x_bound 6 --xdot_bound 5 --thetadot_bound 5 \
  --set_mode avoid \
  --data_root /path/to/cartpole_pybullet \
  --training_objective temporal_consistency \
  --tc_loss_weight 1.0 \
  --tc_t0_mode weighted \
  --tc_anchor_weight 0.1 \
  --num_supervised 128 \
  --supervised_labels_file cal_set.txt \
  --supervised_weight 1.0 \
  --trajectory_uniform_sampling \
  --numpoints 2000 \
  --num_epochs 200 \
  --lr 1e-4 \
  --num_hl 2 --num_nl 128 \
  --model sine
```

**Training Logs:**
Temporal consistency mode logs the following metrics:
- `tc_backup`: Temporal PDE residual loss `|dV/dt + ∇V·ẋ_obs|²`
- `tc_anchor`: Boundary loss at `t=tMin` (if `tc_t0_mode != 'off'`)
- `tc_total`: Combined temporal loss

**Notes:**
- `training_objective=hj_pde` remains the default and preserves previous behavior
- `training_objective=temporal_consistency` currently supports `CartPoleDataset` only
- Temporal loss uses observed flow `ẋ_obs` computed from one-step trajectory differences
- `--tc_t0_mode {weighted,fixed,off}` controls whether/how `t=tMin` boundary loss is applied
- For implementation-level details, see `MEGA_NOTEBOOK.md`

### Using Configuration Files

For convenience, several configuration files are provided in `configs/`:

**Available Configs:**
- `cartpole_50k_70k.cfg` - Large-scale training with 50k pretrain + 70k finetune curriculum
- `cartpole_large_X_calset.cfg` - Full dataset with supervised labels
- `cartpole_large_X_calset_15k_6k.cfg` - Medium-scale training (15k pretrain + 6k finetune)
- `cartpole_large_X_calset_15k_6k_trainsplit_ram.cfg` - Training split with RAM loading

**Usage:**
```bash
python run_experiment.py -c configs/cartpole_50k_70k.cfg --mode train
```

**Shell Scripts:**
Convenience scripts are also provided:
```bash
./run_cartpole_50k_70k.sh
./run_cartpole_large_X_calset.sh
./run_cartpole_large_X_calset_15k_6k.sh
./run_cartpole_15k_6k_trainsplit_ram.sh
```

## Evaluation

### ROA (Region of Attraction) Evaluation

The `evaluation/eval_roa.py` script evaluates learned value functions for ROA prediction with comprehensive metrics.

**Evaluation Workflow:**

1. **Calibrate threshold** on `cal_set.txt` to find optimal decision boundary
2. **Evaluate** on `test_set.txt` with the calibrated threshold
3. Optionally set `--separatrix_margin M` to create a no-decision band `[threshold-M, threshold+M]` and track coverage

**Example: Basic evaluation**
```bash
python evaluation/eval_roa.py \
  --experiment_dir runs/cartpole_small \
  --checkpoint model_final.pth \
  --cal_set cartpole_pybullet/cal_set.txt \
  --test_set cartpole_pybullet/test_set.txt \
  --t_eval 2.0 \
  --auto_threshold \
  --optimize_metric f1 \
  --separatrix_margin 0.0 \
  --threshold_steps 1001
```

**Example: Evaluation at specific timestamp**
```bash
python evaluation/eval_roa.py \
  --experiment_dir runs/cartpole_50k_70k_trainsplit_ram_613ts \
  --checkpoint model_final.pth \
  --cal_set cartpole_pybullet/cal_set.txt \
  --test_set cartpole_pybullet/test_set.txt \
  --timestamp_index 613 \
  --timestamp_dt 0.01 \
  --auto_threshold \
  --optimize_metric f1
```

**Evaluation Features:**
- **Metrics**: Reports precision, recall, F1, accuracy, balanced accuracy, and specificity
- **Threshold Optimization**: Automatically finds optimal threshold on calibration set
- **Timestamp Support**: Evaluate at discrete timestamps via `--timestamp_index` (e.g., `613`), converted using `t_eval = tMin + timestamp_index * dt`
- **Separatrix Margin**: Creates no-decision region around threshold for uncertainty quantification

**Evaluation Parameters:**
- `--experiment_dir`: Directory containing trained model checkpoints
- `--checkpoint`: Model checkpoint file (e.g., `model_final.pth`)
- `--cal_set`: Calibration set for threshold tuning
- `--test_set`: Test set for final evaluation
- `--t_eval`: Evaluation time (or use `--timestamp_index` with `--timestamp_dt`)
- `--auto_threshold`: Automatically find optimal threshold
- `--optimize_metric`: Metric to optimize (`f1`, `accuracy`, `balanced_accuracy`, etc.)
- `--separatrix_margin`: Margin around threshold for no-decision region
- `--threshold_steps`: Number of threshold candidates to test

## Example Experiments

### Experiment 1: HJ PDE with Supervised Labels (50k + 70k Curriculum)

**Experiment Directory:** `runs/cartpole_50k_70k_trainsplit_ram_613ts`

**Configuration:**
- Training objective: `hj_pde` (default)
- Dataset: Training split only (`train_test_splits/shuffled_indices.txt`)
- Trajectories: Preloaded in RAM for faster access
- Time horizon: `tMax=6.13` (613 timesteps at dt=0.01)
- Curriculum: 50k pretrain iterations + 70k finetune iterations
- Supervised labels: `cal_set.txt` with 128 samples per batch

**Run Command:**
```bash
python run_experiment.py -c configs/cartpole_50k_70k.cfg --mode train
# Or modify tMax in config:
python run_experiment.py -c configs/cartpole_50k_70k.cfg --mode train --tMax 6.13
```

**Evaluation:**
```bash
python evaluation/eval_roa.py \
  --experiment_dir runs/cartpole_50k_70k_trainsplit_ram_613ts \
  --checkpoint model_final.pth \
  --cal_set cartpole_pybullet/cal_set.txt \
  --test_set cartpole_pybullet/test_set.txt \
  --timestamp_index 613 \
  --timestamp_dt 0.01 \
  --auto_threshold \
  --optimize_metric f1
```

### Experiment 2: Temporal Consistency (Weighted t0 Boundary)

**Experiment Directory:** `runs/cartpole_pybullet_temporal_v1_nosup_ram_t0anchor`

**Configuration:**
- Training objective: `temporal_consistency`
- Boundary mode: `weighted` (t0 boundary loss with 0.1 weight)
- Temporal loss weight: 1.0
- No supervised labels (pure temporal consistency)
- Trajectories: Preloaded in RAM
- Time horizon: `tMax=6.13`

**Run Command:**
```bash
python run_experiment.py --mode train \
  --experiment_name cartpole_pybullet_temporal_v1_nosup_ram_t0anchor \
  --experiment_class DeepReach \
  --dynamics_class CartPole \
  --minWith target \
  --u_max 2000 --x_bound 6 --xdot_bound 5 --thetadot_bound 5 \
  --set_mode avoid \
  --data_root /path/to/cartpole_pybullet \
  --dataset_class CartPoleDataset \
  --training_objective temporal_consistency \
  --deepreach_model exact \
  --tc_loss_weight 1.0 \
  --tc_t0_mode weighted \
  --tc_anchor_weight 0.1 \
  --trajectory_uniform_sampling \
  --load_trajectories_in_ram \
  --tMin 0.0 --tMax 6.13 \
  --numpoints 2000 \
  --num_epochs 40000 \
  --lr 7e-5 \
  --num_hl 2 --num_nl 96 \
  --model sine \
  --steps_til_summary 2000 \
  --epochs_til_ckpt 5000 \
  --device cuda:0 \
  --seed 0
```

### Experiment 3: Temporal Consistency (Fixed t0 Boundary)

**Experiment Directory:** `runs/cartpole_pybullet_temporal_v1_nosup_ram_t0ancho_tightBoundary`

**Configuration:**
- Training objective: `temporal_consistency`
- Boundary mode: `fixed` (t0 boundary loss with weight 1.0)
- Temporal loss weight: 1.0
- No supervised labels
- Trajectories: Preloaded in RAM
- Time horizon: `tMax=6.13`

**Run Command:**
```bash
python run_experiment.py --mode train \
  --experiment_name cartpole_pybullet_temporal_v1_nosup_ram_t0ancho_tightBoundary \
  --experiment_class DeepReach \
  --dynamics_class CartPole \
  --minWith target \
  --u_max 2000 --x_bound 6 --xdot_bound 5 --thetadot_bound 5 \
  --set_mode avoid \
  --data_root /path/to/cartpole_pybullet \
  --dataset_class CartPoleDataset \
  --training_objective temporal_consistency \
  --deepreach_model exact \
  --tc_loss_weight 1.0 \
  --tc_t0_mode fixed \
  --tc_anchor_weight 0.1 \
  --trajectory_uniform_sampling \
  --load_trajectories_in_ram \
  --tMin 0.0 --tMax 6.13 \
  --numpoints 2000 \
  --num_epochs 40000 \
  --lr 7e-5 \
  --num_hl 2 --num_nl 96 \
  --model sine \
  --device cuda:0 \
  --seed 0
```

## Monitoring a DeepReach Experiment
Results for the Dubins3D system specified in the above section can be found in this [online WandB project](https://wandb.ai/aklin/DeepReachTutorial).
We highly recommend users use the `--use_wandb` flag to log training progress to the free cloud-based Weights & Biases AI Developer Platform, where it can be easily viewed and shared.

Throughout training, the training loss curves, value function plots, and model checkpoints are saved locally to `runs/experiment_name/training/summaries` and `runs/experiment_name/training/checkpoints` (and to WandB, if specified).

## Inspecting Pretraining Curriculum

The `scripts/inspect_pretrain_curriculum.py` script allows you to inspect pretraining progress and curriculum behavior:

```bash
python scripts/inspect_pretrain_curriculum.py \
  --experiment_dir runs/cartpole_50k_70k_trainsplit_ram_613ts \
  --checkpoint model_final.pth \
  --training_objective hj_pde \
  --num_samples 1000
```

For temporal consistency mode:
```bash
python scripts/inspect_pretrain_curriculum.py \
  --experiment_dir runs/cartpole_temporal_weighted \
  --checkpoint model_final.pth \
  --training_objective temporal_consistency \
  --tc_t0_mode weighted \
  --num_samples 1000
```

## Defining a Custom System
Systems are defined in `dynamics/dynamics.py` and inherit from the abstract `Dynamics` class. At a minimum, users must define:
* `__init(self, ...)__`, which must call `super().__init__(loss_type, set_mode, state_dim, ...)`
* `state_test_range(self)`, which specifies the state space that will be visualized in training plots
* `dsdt(self, state, control, disturbance)`, which implements the forward dynamics
* `boundary_fn(self, state)`,  which implements the boundary function that implicitly represents the target set
* `hamiltonian(self, state, dvds)`, which implements the system's hamiltonian
* `plot_config(self)`, which specifies the state slices and axes visualized in training plots

## Citation
If you find our work useful in your research, please cite:
```
@software{deepreach2024,
  author = {Lin, Albert and Feng, Zeyuan and Borquez, Javier and Bansal, Somil},
  title = {{DeepReach Repository}},
  url = {https://github.com/smlbansal/deepreach},
  year = {2024}
}
```

```
@inproceedings{bansal2020deepreach,
    author = {Bansal, Somil
              and Tomlin, Claire},
    title = {{DeepReach}: A Deep Learning Approach to High-Dimensional Reachability},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year={2021}
}
```

## Contact
If you have any questions, please feel free to email the authors.
