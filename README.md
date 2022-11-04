# S4'mer

[![black](https://github.com/Dana-Farber-AIOS/s4mer/actions/workflows/black.yml/badge.svg)](https://github.com/Dana-Farber-AIOS/s4mer/actions/workflows/black.yml)

| Branch | Test status   |
| ------ | ------------- |
| master | ![tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=master) |
| dev    | ![tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=dev) |

TODO: Change banners from pathML's to VS4's

This repo implements and benchmarks a number of drop-in replacements for attention:

- [HiPPO](https://arxiv.org/pdf/2008.07669.pdf)
- [HtTYH](https://arxiv.org/pdf/2206.12037.pdf)
- [DSS](https://arxiv.org/pdf/2203.14343.pdf)
- [GSS](https://arxiv.org/pdf/2206.13947.pdf)
- [S4](https://arxiv.org/pdf/2111.00396.pdf)
- [S4D](https://arxiv.org/pdf/2206.11893.pdf)

The data we used to benchmark this is:

- [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Camelyon17](https://camelyon17.grand-challenge.org/)
- [TIGER](https://tiger.grand-challenge.org/)
- [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

**View [documentation](https://pathml.readthedocs.io/en/latest/)**

:construction: the `dev` branch is under active development, with experimental features, bug fixes, and refactors that may happen at any time!
Stable versions are available as tagged releases on GitHub, or as versioned releases on PyPI

# Installation

Install [poetry](https://python-poetry.org/):

```bash
curl -sSL https://install.python-poetry.org | python3 - --preview
```

Ensure python version is set to 3.8:

```bash
$ python --version
> 3.8.x
```

Install lock file dependencies:

```bash
poetry install --with jax,torch,mltools
```

if you plan to use a jupyter notebook also install:

```bash
poetry install --with jupyter
```

Thats it!

## Dev

## Installation

Install [poetry](https://python-poetry.org/):

```bash
curl -sSL https://install.python-poetry.org | python3 - --preview
```

Ensure python version is set to 3.8:

```bash
$ python --version
> 3.8.13
```

Install lock file dependencies:

```bash
poetry install --with torch,jax,mltools,additional
```

if you plan to use a jupyter notebook:

```bash
poetry install --with jupyter
```

Refer to [this](https://python-poetry.org/docs/cli#add) in the future if you need help

```bash
poetry add --editable ../s4mer/
```

There are several ways to run your own local `S4'former` experiments:

TODO: add installation guide

# Contributing

TODO: add contributing guide

# Citing

TODO: add citation guide

# License

The GNU GPL v2 version of PathML is made available via Open Source licensing.
The user is free to use, modify, and distribute under the terms of the GNU General Public License version 2.

Commercial license options are available also.

# Contact

Questions? Comments? Suggestions? Get in touch!

[bryan_gass@dfci.harvard.edu](mailto:bryan_gass@dfci.harvard.edu)

<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/dfci_cornell_joint_logos.png width="750">
