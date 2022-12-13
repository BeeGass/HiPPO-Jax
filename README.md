# S4'mer
<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Typical_State_Space_model.svg/472px-Typical_State_Space_model.svg.png width=600>

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
- [S5](https://arxiv.org/abs/2208.04933)
- [Mega](https://arxiv.org/abs/2209.10655)
- [H3 Block](https://openreview.net/pdf?id=COZDy0WYGg)

The data we used to benchmark this is:

- [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Camelyon17](https://camelyon17.grand-challenge.org/)
- [TIGER](https://tiger.grand-challenge.org/)
- [PanNuke](https://jgamper.github.io/PanNukeDataset/)
- [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [PanNuke](https://jgamper.github.io/PanNukeDataset/)
- [Mesmer](https://www.biorxiv.org/content/10.1101/2021.03.01.431313v2.full.pdf)
- [TCGA](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga)

**View [documentation](https://pathml.readthedocs.io/en/latest/)**

:construction: the `dev` branch is under active development, with experimental features, bug fixes, and refactors that may happen at any time!
Stable versions are available as tagged releases on GitHub, or as versioned releases on PyPI

## Installation
---
There are several ways to install S4mer:

1. Use a package manager
    1. poetry install (recommended for users)
    2. pip install from PyPI
2. Clone repo to local machine and install from source (recommended for developers/contributors)

Ensure your [CUDA drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installation) have been installed correctly, this will effect dependencies like [Jax](https://github.com/google/jax#installation) and [PyTorch](https://pytorch.org/)

Note: these instructions are for Linux. Commands may be different for other platforms.

### Installation option 1: poetry install
---

1. Install [poetry](https://python-poetry.org/):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Ensure python version is set to 3.8:

```bash
$ python --version
> 3.8.x
```

3. Activate poetry virtual environment

```bash
poetry shell
```

4. (optional) Update the dependencies to ensure dependencies work with your system

```bash
poetry update
```

5. Install lock file dependencies:

```bash
poetry install --with jax,torch,mltools,jupyter,additional,dataset
```

### Installation option 1: pip install
---

1. Create and activate virtual environment
```bash
conda create --name s4mer python=3.8
conda activate s4mer
```

2. Install dependencies 
```bash
pip install -r requirements.txt
```

### Installation option 2: clone repo and install from source
---

1. Clone repo:

via HTTPS:
```bash
git clone https://github.com/Dana-Farber-AIOS/s4mer.git
cd s4mer
```

via SSH
```bash
git clone git@github.com:Dana-Farber-AIOS/s4mer.git
cd s4mer
```

2. Create conda environment:

```bash
conda env create -f requirements.txt
conda activate s4mer
```

3. Install `S4mer` from source:

```bash
pip install -e .
```

Thats it!

## Examples

```python
import jax.random as jr

key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
```

**HiPPO**

```python
from s4mer.src.models.hippo.transition import TransMatrix
from s4mer.src.models.hippo.hippo import HiPPO

N = 100
gbt_type = 0.5
L = 784 # length of data
measure = "legs"

matrices = TransMatrix(N, measure)
A = matrices.A_matrix
B = matrices.B_matrix

hippo = HiPPO(
    N, L, 1.0 / L, gbt_type, L, A, B, measure
)


```

# Running Experiments

There are several ways to run your own local `S4'former` experiments:

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

[bagass@wpi.edu](mailto:bagass@wpi.edu)

<p align="center"> 
<img style="vertical-align:middle" src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/dfci_cornell_joint_logos.png width="525"> 
<img style="vertical-align:middle" src=https://www.wpi.edu/sites/default/files/inline-image/Offices/Marketing-Communications/WPI_Inst_Prim_FulClr_PREVIEW.png?1670371200029 width=200>
</p> 