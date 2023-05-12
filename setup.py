# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': '.'}

packages = \
['src',
 'src.data',
 'src.data.dataloaders',
 'src.datasets',
 'src.loss',
 'src.models',
 'src.models.hippo',
 'src.optimizer',
 'src.pipeline',
 'src.tests',
 'src.tests.hippo_tests',
 'src.train',
 'src.utils']

package_data = \
{'': ['*'],
 'src': ['configs/*',
         'configs/dataset/*',
         'configs/loss/*',
         'configs/model/*',
         'configs/optimizer/*',
         'configs/pipeline/*',
         'configs/task/*']}

install_requires = \
['jaxtyping>=0.2.11,<0.3.0']

setup_kwargs = {
    'name': 'hippo-pkg',
    'version': '1.0.0',
    'description': 'Repo for the testing and developing of state space models, transformers and the hybrid between the two; trained on whole slide digital pathology image data.',
    'long_description': '# HiPPO-Jax\n\n<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Typical_State_Space_model.svg/472px-Typical_State_Space_model.svg.png width=600>\n\nThis repo uses ideas and code that can be both found at [HazyResearch/state-spaces](https://github.com/HazyResearch/state-spaces). This code base implements the ideas and code in jax. \n\n## Installation\n\nThere are several ways to install HiPPO-Jax:\n\n1. Use a package manager\n    1. poetry install (recommended for users)\n    2. pip install from PyPI\n2. Clone repo to local machine and install from source (recommended for developers/contributors)\n\nEnsure your [CUDA drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installation) have been installed correctly, this will effect dependencies like [Jax](https://github.com/google/jax#installation) and [PyTorch](https://pytorch.org/)\n\nNote: these instructions are for Linux. Commands may be different for other platforms.\n\n### Installation option 1: poetry install\n\n---\n\n1. Install [poetry](https://python-poetry.org/):\n\n```bash\ncurl -sSL https://install.python-poetry.org | python3 -\n```\n\n2. Ensure python version is set to 3.8:\n\n```bash\n$ python --version\n> 3.8.x\n```\n\n3. Activate poetry virtual environment\n\n```bash\npoetry shell\n```\n\n4. (optional) Update the dependencies to ensure dependencies work with your system\n\n```bash\npoetry update\n```\n\n5. Install lock file dependencies:\n\n```bash\npoetry install --with jax,torch,mltools,jupyter,additional,dataset\n```\n\n### Installation option 1: pip install\n\n---\n\n1. Create and activate virtual environment\n\n```bash\nconda create --name hippo_jax python=3.8\nconda activate hippo_jax\n```\n\n2. Install dependencies\n\n```bash\npip install -r requirements.txt\n```\n\n### Installation option 2: clone repo and install from source\n\n---\n\n1. Clone repo:\n\nvia HTTPS:\n\n```bash\ngit clone https://github.com/Dana-Farber-AIOS/HiPPO-Jax.git\ncd HiPPO-Jax\n```\n\nvia SSH\n\n```bash\ngit clone git@github.com:Dana-Farber-AIOS/HiPPO-Jax.git\ncd HiPPO-Jax\n```\n\n2. Create conda environment:\n\n```bash\nconda env create -f requirements.txt\nconda activate hippo_jax\n```\n\n3. Install `Hippo-Jax` from source:\n\n```bash\npip install -e .\n```\n\nThats it!\n\n## Examples\n\n```python\nimport jax.random as jr\n\nkey, subkey = jr.split(jr.PRNGKey(0), 2)\n```\n\n**HiPPO Matrices**\n\n```python\nfrom src.models.hippo.transition import TransMatrix\n\nN = 100\nmeasure = "legs"\n\nmatrices = TransMatrix(N=N, measure=measure)\nA = matrices.A\nB = matrices.B\n```\n\n**HiPPO (LTI) Operator**\n\n```python\nfrom src.models.hippo.hippo import HiPPOLTI\n\nN = 50\nT = 3\nstep = 1e-3\nmeasure = "legs"\ndesc_val = 0.0\n\nhippo = HiPPOLTI(\n        N=N,\n        step_size=step,\n        GBT_alpha=desc_val,\n        measure=measure,\n        basis_size=T,\n        unroll=False,\n    )\n\n```\n\n**HiPPO (LSI) Operator**\n\n```python\nfrom src.models.hippo.hippo import HiPPOLSI\n\nN = 50\nT = 3\nstep = 1e-3\nL = int(T / step)\nmeasure = "legs"\ndesc_val = 0.0\n\nhippo = HiPPOLSI(\n        N=N,\n        max_length=L,\n        step_size=step,\n        GBT_alpha=desc_val,\n        measure=measure,\n        unroll=True,\n    )\n\n```\n\n**Use right out of the box, no training needed**\n\n```python\nparams = hippo.init(key, f=x)\nc, y = hippo.apply(params, f=x)\n```\n\n# Contributing\n\n`HiPPO-Jax` is an open source project. Consider contributing to benefit the entire community!\n\nThere are many ways to contribute to `HiPPO-Jax`, including:\n\n- Submitting bug reports\n- Submitting feature requests\n- Writing documentation and examples\n- Fixing bugs\n- Writing code for new features\n- Sharing workflows\n- Sharing trained model parameters\n- Sharing `HiPPO-Jax` with colleagues, students, etc.\n\n# License\n\nThe GNU GPL v2 version of HiPPO-Jax is made available via Open Source licensing.\nThe user is free to use, modify, and distribute under the terms of the GNU General Public License version 2.\n\nCommercial license options are available also.\n\n# Contact\n\nQuestions? Comments? Suggestions? Get in touch!\n\n[bagass@wpi.edu](mailto:bagass@wpi.edu)\n\n<p align="center">\n<img style="vertical-align:middle" src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/dfci_cornell_joint_logos.png width="525">\n<img style="vertical-align:middle" src=https://www.wpi.edu/sites/default/files/inline-image/Offices/Marketing-Communications/WPI_Inst_Prim_FulClr_PREVIEW.png?1670371200029 width=200>\n</p>\n',
    'author': 'Bryan Gass',
    'author_email': 'bryank123@live.com',
    'maintainer': 'Bryan Gass',
    'maintainer_email': 'bryank123@live.com',
    'url': 'https://github.com/Dana-Farber-AIOS/HiPPO-Jax',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8.0,<=3.8.13',
}


setup(**setup_kwargs)

