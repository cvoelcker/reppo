# Relative Entropy Pathwise Policy Optimization 

> [!NOTE]
> We are building more accessible versions of REPPO (meaning they are in Torch ;) ) that can be integrated with well-established RL frameworks. To kick things off, take a look at our rsl_rl implementation:
> * https://github.com/cvoelcker/rsl_rl: RSL RL is a popular framework that integrates well with IsaacLab and other robotics simulators

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

The main code for the algorithm is in `src/jaxrl/reppo.py` and `src/torchrl/reppo.py` respectively.
In our tests, both versions produce similar returns up to seed variance.
However, due to slight variations in the frameworks, we cannot always guarantee this.

For maximum speed, we highly recommend using our jax version.
The torch version can result in slow experiment depending on the CPU/GPU configuration, as sampling from a squashed Gaussian is not implemented efficiently in the torch framework.
This can result in cases where the GPU is stalled if the CPU cannot provide instructions and kernels fast enough.

Our configurations are handled with [hydra.cc](https://hydra.cc/). This means parameters can be overwritten by using the syntax
```bash
python src/jaxrl/reppo.py PARAMETER=VALUE
```

By default, the environment type and name need to be provided.
Currently the jax version supports `env=mjx_dmc`, `env=mjx_humanoid`, `env=brax`, and `env=humanoid_brax`. The latter is treated as a separate environment, as the reward scale is much larger than other brax environments, and the min and max Q values need to be tracked per environment.
The torch version support `env=mjx_dmc`, and `env=maniskill`. We additionally provide wrappers for isaaclab, but this is still under development and might not work out of the box.

The paper experiments can be reproduced easily by using the `experiment_override` settings.
By specifying `experiment_override=mjx_smc_small_data` for example, you can run the variant of REPPO with a batch size of 32k samples.

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
