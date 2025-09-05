# Relative Entropy Pathwise Policy Optimization 

## On-policy value-based reinforcement learning without endless hyperparameter tuning

This repository contains the official implementation for REPPO - Relative Entropy Pathwise Policy Optimization [arXiv paper link](https://arxiv.org/abs/2507.11019).

We provide reference implementations of the REPPO algorithm, as well as the raw results for our experiments.

Our repo provides you with the core algorithm and the following features:
- Jax and Torch support: No matter what your favorite framework is, you can take use the algorithm out of the box
- Modern installation: Our algorithm and environment dependencies can be installed with a single command
- Fast and reliable learning: REPPO is wallclock time competitive with approaches such as FastTD3 and PPO, while learning reliably and with minimal hyperparameter tuning

## Installation

We strongly recommend using the [uv tool](https://docs.astral.sh/uv/getting-started/installation/) for python dependency management.

With uv installed, you can install the project and all dependencies in a local virtual environment under `.venv` with one single command:
```bash 
uv sync
```

Our installation requires a GPU with CUDA 12 compatible drivers.

If you use other dependency management tools such as conda, create a new environment with `Python 3.12` and install our package with
```bash
pip install -e .
```

> [!Note]
> Several mujoco_playground environments, such as the Humanoid tasks, are currently unstable. If environments result in nans, we have simply rerun our experiments manually. As soon as these issues are solved upstream, we will update our dependencies.

> [!NOTE]
>  To provide a level comparison with prior work, we depend on the FastTD3 for of mujoco_playground. As soon as proper terminal state observation handling is merged into the main repository, we will update our dependencies.


## Running Experiments

The main code for the algorithm is in `src/algorithms/reppo`.
Our configurations are handled with [hydra.cc](https://hydra.cc/). This means parameters can be overwritten by using the syntax
```bash
python src/algorithms/reppo/train_reppo.py PARAMETER_NAME=VALUE
```

Algorithms are independent of the environment, and can be run with either [gymnax](https://github.com/google/gymnax) or [gymnasium](https://github.com/Farama-Foundation/Gymnasium) style environments.
To launch a training run, use the following command:
```bash
python src/algorithms/reppo/train_reppo.py --config-name ff_playground.yaml env.name=CartpoleBalance
```

Algorithm configurations have the following structure:
```
config/
  algorithm/
  runner/
  env/
  networks/
  logging/
```
The `algorithm` folder contains the core algorithm hyperparameters, such as learning rates, batch sizes, etc.
The `runner` folder contains the environment interaction configurations, such as the type of rollout collection logic, number of steps per rollout, etc.
The `env` folder contains environment specific parameters, such as the environment name and framework.
The `networks` folder contains the neural network architectures. For instance, we provide configurations for continuous and discrete action spaces with feedforward networks for REPPO (`reppo_continuous`, `reppo_discrete`).
The `logging` folder contains logging configurations, such as whether to log to wandb.
For instance, to run REPPO on the gymnasium discrete action Cartpole-v1 environment, you can use the following command:
```bash
python src/algorithms/reppo/train_reppo.py algorithm=reppo runner=gymnasium env=gymnasium env.name=CartPole-v1 networks=reppo_discrete
```

Default configurations for an algorithm can be found in `config/default/ALGORITHM_NAME`.
Those configurations can be specified with the `--config-name` flag, and can be used to run experiments for different environments quickly.
Default configurations can be overwritten by specifying experiment specific overrides using hydras append feature:
```bash
python src/algorithms/reppo/train_reppo.py --config-name ff_playground.yaml +experiments=mjx_dmc_small_data env.name=CartpoleBalance
```

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details. The repository is built on prior code from the [PureJaxRL](https://github.com/luchris429/purejaxrl) and [FastTD3](https://github.com/younggyoseo/FastTD3) projects, and we thank the respective authors for making their work available in open-source. We include the appropriate licences in ours.

## Citation

```bibtex
@article{voelcker2025reppo,
  title     = {Relative Entropy Pathwise Policy Optimization},
  author    = {Voelcker, Claas and Brunnbauer, Axel and Hussing, Marcel and Nauman, Michal and Abbeel, Pieter and Eaton, Eric and Grosu, Radu and Farahmand, Amir-massoud and Gilitschenski, Igor},
  booktitle = {preprint},
  year      = {2025},
}
```
