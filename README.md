# DePO codes

Official Pytorch implemetation of ICML2022 paper [Depo (Plan Your Target and Learn Your Skills: Transferable State-Only Imitation Learning via Decoupled Policy Optimization)](https://arxiv.org/abs/2203.02214).

**Important Notes**

This repository is based on [ILSwiss](https://github.com/Ericonaldo/ILSwiss). The code is for Mujoco experiments, if you are looking for NGSIM experiments, check [here](https://github.com/apexrl/DePO_NGSIM).

# Algorithms Contained

Implemented RL algorithms:

- Soft-Actor-Critic (SAC)

Implemented LfD algorithms:

- Adversarial methods for Inverse Reinforcement Learning
  - AIRL / GAIL / FAIRL / DAC
- BC

Implemented LfO algorithms:

- BCO
- GAIfO
- DPO

# Running Notes:

Before running, assign important log and output paths in `\rlkit\launchers\config.py`.

There are simple multiple processing shcheduling (we use multiple processing to clarify it with multi-processing since it only starts many independent sub-process without communication) for simple hyperparameter grid search.

The main entry is `run_experiments.py`, with the assigned experiment yaml file in `\exp_specs`:
`python run_experiment.py -g 0 -e your_yaml_path` or `python run_experiment.py -e your_yaml_path`.

When you run the `run_experiments.py`, it reads the yaml file, and generate small yaml files with only one hyperparameter setting for each. In a yaml file, a script file path is assigned (see `\run_scripts\`), which is specified to run the script with every the small yaml file. See `\exp_specs\sac\bc.yaml` for necessary explaination of each parameter.

NOTE: all experiments, including the evaluation tasks (see `\run_scripts\evaluate_policy.py` and `\exp_specs\evaluate_policy`) and the render tasks, can be run under this framework by specifying the yaml file (in a multiple processes style).

## Reproducing Results

### Training Expert Policies

Train an SAC agent and collect expert demos, or use the demo [here](https://github.com/apexrl/Baseline_Pool/tree/master/imitation_learning/sac/expert_trajs_50). Then write the demon path in `\demos_listing.yaml`.

### Example scripts

`-e` means the path to the yaml file, `-g` means gpu id. Existing specs are the ones for producing the final results.

#### Training LfO Agents

**Config files are in `exp_specs/dpo_exps`. Example commands**:

BCO

```bash
python run_experiment.py --nosrun -e exp_specs/dpo_exps/bco_hopper_4.yaml
```

GAIfO

```bash
python run_experiment.py --nosrun -e exp_specs/dpo_exps/gailfo_hopper_4.yaml
```

GAIfO-DP

```bash
python run_experiment.py --nosrun -e exp_specs/dpo_exps/gailfo_dp_hopper_4.yaml
```

DPO (Supervised)

```bash
python run_experiment.py --nosrun -e exp_specs/dpo_exps/sl_lfo_hopper_4.yaml
```

DPO

```bash
python run_experiment.py --nosrun -e exp_specs/dpo_exps/dpo_hopper_4_weightedmle_qsa_weight.yaml
```

#### Abaltion Study

**Config files are in `exp_specs/ablation`. Example commands**:

```bash
python run_experiment.py --nosrun -e exp_specs/ablation/dpo_hopper_4_weightedmle_qsa_static_lambdah.yaml
```

#### Transfer Experiments

**Config files are in `exp_specs/transfer_exps` and `exp_specs/complex_transfer`. Example commands** (remember to change the loaded policy ckpt path in the yaml file):

```bash
python run_experiment.py --nosrun -e exp_specs/transfer_exps/dpo_hopper_4_weightedmle_qsa_weight.yaml
```

#### RL Experiments

**Config files are in `exp_specs/rl`. Example commands**:

```bash
python run_experiment.py --nosrun -e exp_specs/rl/dpo_hopper.yaml
```

#### RL transfer Experiments

**Config files are in `exp_specs/rl_transfer`. Example commands** (remember to change the loaded policy ckpt path in the yaml file):

```bash
python run_experiment.py --nosrun -e exp_specs/rl_transfer/dpo_hopper.yaml
```

### Evaluate state planner

**Config files are in `exp_specs/evaluation`. Example commands**  (remember to change the loaded policy ckpt in `evaluate_state_predictor.py`):

```bash
python run_experiment.py --nosrun -e exp_specs/evaluation/eval_sp.yaml
```
