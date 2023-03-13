# HiPPO-Jax

<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Typical_State_Space_model.svg/472px-Typical_State_Space_model.svg.png width=600>

This repo uses ideas and code that can be both found at [HazyResearch/state-spaces](https://github.com/HazyResearch/state-spaces). This code base implements the ideas and code in jax. 

## Installation

There are several ways to install HiPPO-Jax:

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
conda create --name hippo_jax python=3.8
conda activate hippo_jax
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
git clone https://github.com/Dana-Farber-AIOS/HiPPO-Jax.git
cd HiPPO-Jax
```

via SSH

```bash
git clone git@github.com:Dana-Farber-AIOS/HiPPO-Jax.git
cd HiPPO-Jax
```

2. Create conda environment:

```bash
conda env create -f requirements.txt
conda activate hippo_jax
```

3. Install `Hippo-Jax` from source:

```bash
pip install -e .
```

Thats it!

## Examples

```python
import jax.random as jr

key, subkey = jr.split(jr.PRNGKey(0), 2)
```

**HiPPO Matrices**

```python
from hippo_jax.src.models.hippo.transition import TransMatrix

N = 100
measure = "legs"

matrices = TransMatrix(N=N, measure=measure)
A = matrices.A
B = matrices.B
```

**HiPPO (LTI) Operator**

```python
from hippo_jax.src.models.hippo.hippo import HiPPOLTI

N = 50
T = 3
step = 1e-3
measure = "legs"
desc_val = 0.0

hippo = HiPPOLTI(
        N=N,
        step_size=step,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        unroll=False,
    )

```

**HiPPO (LTS) Operator**

```python
from hippo_jax.src.models.hippo.hippo import HiPPOLSI

N = 50
T = 3
step = 1e-3
L = int(T / step)
measure = "legs"
desc_val = 0.0

hippo = HiPPOLSI(
        N=N,
        max_length=L,
        step_size=step,
        GBT_alpha=desc_val,
        measure=measure,
        unroll=True,
    )

```

**Use right out of the box, no training needed**

```python
params = hippo.init(key, f=x)
c = hippo.apply(params, f=x)
```

# Contributing

`HiPPO-Jax` is an open source project. Consider contributing to benefit the entire community!

There are many ways to contribute to `HiPPO-Jax`, including:

- Submitting bug reports
- Submitting feature requests
- Writing documentation and examples
- Fixing bugs
- Writing code for new features
- Sharing workflows
- Sharing trained model parameters
- Sharing `HiPPO-Jax` with colleagues, students, etc.

# License

The GNU GPL v2 version of HiPPO-Jax is made available via Open Source licensing.
The user is free to use, modify, and distribute under the terms of the GNU General Public License version 2.

Commercial license options are available also.

# Contact

Questions? Comments? Suggestions? Get in touch!

[bagass@wpi.edu](mailto:bagass@wpi.edu)

<p align="center">
<img style="vertical-align:middle" src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/dfci_cornell_joint_logos.png width="525">
<img style="vertical-align:middle" src=https://www.wpi.edu/sites/default/files/inline-image/Offices/Marketing-Communications/WPI_Inst_Prim_FulClr_PREVIEW.png?1670371200029 width=200>
</p>
