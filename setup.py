# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': '.'}

packages = \
['src',
 'src.config',
 'src.config.data',
 'src.config.models',
 'src.data',
 'src.data.dataloaders',
 'src.models',
 'src.models.hippo',
 'src.models.rnn',
 'src.models.ss',
 'src.optimizer',
 'src.tasks',
 'src.tests',
 'src.tests.hippo_tests',
 'src.tests.rnn_tests',
 'src.utils']

package_data = \
{'': ['*'],
 'src': ['configs/model/hippo/dplr_hippo/*',
         'configs/model/hippo/hippo_lsi/*',
         'configs/model/hippo/hippo_lti/*',
         'configs/model/hippo/nplr_hippo/*',
         'configs/model/s4/*',
         'configs/optimizer/*']}

install_requires = \
['jaxtyping>=0.2.11,<0.3.0']

setup_kwargs = {
    'name': 's4mer-pkg',
    'version': '1.0.0',
    'description': 'Repo for the testing and developing of state space models, transformers and the hybrid between the two; trained on whole slide digital pathology image data.',
    'long_description': '# S4\'mer\n<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Typical_State_Space_model.svg/472px-Typical_State_Space_model.svg.png width=600>\n\n[![black](https://github.com/Dana-Farber-AIOS/s4mer/actions/workflows/black.yml/badge.svg)](https://github.com/Dana-Farber-AIOS/s4mer/actions/workflows/black.yml)\n\n| Branch | Test status   |\n| ------ | ------------- |\n| master | ![tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=master) |\n| dev    | ![tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=dev) |\n\nTODO: Change banners from pathML\'s to VS4\'s\n\nThis repo implements and benchmarks a number of drop-in replacements for attention:\n\n- [HiPPO](https://arxiv.org/pdf/2008.07669.pdf)\n- [HtTYH](https://arxiv.org/pdf/2206.12037.pdf)\n- [DSS](https://arxiv.org/pdf/2203.14343.pdf)\n- [GSS](https://arxiv.org/pdf/2206.13947.pdf)\n- [S4](https://arxiv.org/pdf/2111.00396.pdf)\n- [S4D](https://arxiv.org/pdf/2206.11893.pdf)\n- [S5](https://arxiv.org/abs/2208.04933)\n- [Mega](https://arxiv.org/abs/2209.10655)\n- [H3 Block](https://openreview.net/pdf?id=COZDy0WYGg)\n\nThe data we used to benchmark this is:\n\n- [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)\n- [Camelyon17](https://camelyon17.grand-challenge.org/)\n- [TIGER](https://tiger.grand-challenge.org/)\n- [PanNuke](https://jgamper.github.io/PanNukeDataset/)\n- [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)\n- [PanNuke](https://jgamper.github.io/PanNukeDataset/)\n- [Mesmer](https://www.biorxiv.org/content/10.1101/2021.03.01.431313v2.full.pdf)\n- [TCGA](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga)\n\n**View [documentation](https://pathml.readthedocs.io/en/latest/)**\n\n:construction: the `dev` branch is under active development, with experimental features, bug fixes, and refactors that may happen at any time!\nStable versions are available as tagged releases on GitHub, or as versioned releases on PyPI\n\n## Installation\n---\nThere are several ways to install S4mer:\n\n1. Use a package manager\n    1. poetry install (recommended for users)\n    2. pip install from PyPI\n2. Clone repo to local machine and install from source (recommended for developers/contributors)\n\nEnsure your [CUDA drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installation) have been installed correctly, this will effect dependencies like [Jax](https://github.com/google/jax#installation) and [PyTorch](https://pytorch.org/)\n\nNote: these instructions are for Linux. Commands may be different for other platforms.\n\n### Installation option 1: poetry install\n---\n\n1. Install [poetry](https://python-poetry.org/):\n\n```bash\ncurl -sSL https://install.python-poetry.org | python3 -\n```\n\n2. Ensure python version is set to 3.8:\n\n```bash\n$ python --version\n> 3.8.x\n```\n\n3. Activate poetry virtual environment\n\n```bash\npoetry shell\n```\n\n4. (optional) Update the dependencies to ensure dependencies work with your system\n\n```bash\npoetry update\n```\n\n5. Install lock file dependencies:\n\n```bash\npoetry install --with jax,torch,mltools,jupyter,additional,dataset\n```\n\n### Installation option 1: pip install\n---\n\n1. Create and activate virtual environment\n```bash\nconda create --name s4mer python=3.8\nconda activate s4mer\n```\n\n2. Install dependencies \n```bash\npip install -r requirements.txt\n```\n\n### Installation option 2: clone repo and install from source\n---\n\n1. Clone repo:\n\nvia HTTPS:\n```bash\ngit clone https://github.com/Dana-Farber-AIOS/s4mer.git\ncd s4mer\n```\n\nvia SSH\n```bash\ngit clone git@github.com:Dana-Farber-AIOS/s4mer.git\ncd s4mer\n```\n\n2. Create conda environment:\n\n```bash\nconda env create -f requirements.txt\nconda activate s4mer\n```\n\n3. Install `S4mer` from source:\n\n```bash\npip install -e .\n```\n\nThats it!\n\n## Examples\n\n```python\nimport jax.random as jr\n\nkey1, key2, key3 = jr.split(jr.PRNGKey(0), 3)\n```\n\n**HiPPO Matrices**\n\n```python\nfrom s4mer.src.models.hippo.transition import TransMatrix\n\nN = 100\nmeasure = "legs"\n\nmatrices = TransMatrix(N=N, measure=measure)\nA = matrices.A\nB = matrices.B\n```\n\n**HiPPO Operator**\n\n```python\nfrom s4mer.src.models.hippo.hippo import HiPPO\n\nN = 100\ngbt_type = 0.5\nL = 784 # length of data\nmeasure = "legs"\n\nhippo = HiPPO(\n    max_length=L,\n    step_size=1.0,\n    N=N,\n    GBT_alpha=gbt_type,\n    measure=measure,\n)\n\n```\n\n# Running Experiments\n\nThere are several ways to run your own local `S4\'former` experiments:\n\n# Contributing\n\nTODO: add contributing guide\n\n# Citing\n\nTODO: add citation guide\n\n# License\n\nThe GNU GPL v2 version of PathML is made available via Open Source licensing.\nThe user is free to use, modify, and distribute under the terms of the GNU General Public License version 2.\n\nCommercial license options are available also.\n\n# Contact\n\nQuestions? Comments? Suggestions? Get in touch!\n\n[bagass@wpi.edu](mailto:bagass@wpi.edu)\n\n<p align="center"> \n<img style="vertical-align:middle" src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/dfci_cornell_joint_logos.png width="525"> \n<img style="vertical-align:middle" src=https://www.wpi.edu/sites/default/files/inline-image/Offices/Marketing-Communications/WPI_Inst_Prim_FulClr_PREVIEW.png?1670371200029 width=200>\n</p> ',
    'author': 'Bryan Gass',
    'author_email': 'bryank123@live.com',
    'maintainer': 'Bryan Gass',
    'maintainer_email': 'bryank123@live.com',
    'url': 'https://github.com/Dana-Farber-AIOS/s4mer',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8.0,<=3.8.13',
}


setup(**setup_kwargs)

