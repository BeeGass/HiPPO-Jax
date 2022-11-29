# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': '.'}

packages = \
['src',
 'src.config',
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
{'': ['*']}

setup_kwargs = {
    'name': 's4mer-pkg',
    'version': '0.1.0',
    'description': 'Repo for the testing and developing of state space models, transformers and the hybrid between the two; trained on whole slide digital pathology image data.',
    'long_description': '# S4\'mer\n\n[![black](https://github.com/Dana-Farber-AIOS/s4mer/actions/workflows/black.yml/badge.svg)](https://github.com/Dana-Farber-AIOS/s4mer/actions/workflows/black.yml)\n\n| Branch | Test status   |\n| ------ | ------------- |\n| master | ![tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=master) |\n| dev    | ![tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=dev) |\n\nTODO: Change banners from pathML\'s to VS4\'s\n\nThis repo implements and benchmarks a number of drop-in replacements for attention:\n\n- [HiPPO](https://arxiv.org/pdf/2008.07669.pdf)\n- [HtTYH](https://arxiv.org/pdf/2206.12037.pdf)\n- [DSS](https://arxiv.org/pdf/2203.14343.pdf)\n- [GSS](https://arxiv.org/pdf/2206.13947.pdf)\n- [S4](https://arxiv.org/pdf/2111.00396.pdf)\n- [S4D](https://arxiv.org/pdf/2206.11893.pdf)\n- [S5](https://arxiv.org/abs/2208.04933)\n- [Mega](https://arxiv.org/abs/2209.10655)\n- [H3 Block](https://openreview.net/pdf?id=COZDy0WYGg)\n\nThe data we used to benchmark this is:\n\n- [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)\n- [Camelyon17](https://camelyon17.grand-challenge.org/)\n- [TIGER](https://tiger.grand-challenge.org/)\n- [PanNuke](https://jgamper.github.io/PanNukeDataset/)\n- [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)\n- [PanNuke](https://jgamper.github.io/PanNukeDataset/)\n\n**View [documentation](https://pathml.readthedocs.io/en/latest/)**\n\n:construction: the `dev` branch is under active development, with experimental features, bug fixes, and refactors that may happen at any time!\nStable versions are available as tagged releases on GitHub, or as versioned releases on PyPI\n\n# Installation\n\nInstall [poetry](https://python-poetry.org/):\n\n```bash\ncurl -sSL https://install.python-poetry.org | python3 - --preview\n```\n\nEnsure python version is set to 3.8:\n\n```bash\n$ python --version\n> 3.8.x\n```\n\nInstall lock file dependencies:\n\n```bash\npoetry install --with jax,torch,mltools,dataset\n```\n\nif you plan to use a jupyter notebook also install:\n\n```bash\npoetry install --with jupyter\n```\n\nThats it!\n\n## Dev\n\n## Installation\n\nInstall [poetry](https://python-poetry.org/):\n\n```bash\ncurl -sSL https://install.python-poetry.org | python3 - --preview\n```\n\nEnsure python version is set to 3.8:\n\n```bash\n$ python --version\n> 3.8.13\n```\n\nInstall lock file dependencies:\n\n```bash\npoetry install --with torch,jax,mltools,additional,dataset\n```\n\nif you plan to use a jupyter notebook:\n\n```bash\npoetry install --with jupyter\n```\n\nRefer to [this](https://python-poetry.org/docs/cli#add) in the future if you need help\n\n```bash\npoetry add --editable ../s4mer/\n```\n\nThere are several ways to run your own local `S4\'former` experiments:\n\nTODO: add installation guide\n\n# Contributing\n\nTODO: add contributing guide\n\n# Citing\n\nTODO: add citation guide\n\n# License\n\nThe GNU GPL v2 version of PathML is made available via Open Source licensing.\nThe user is free to use, modify, and distribute under the terms of the GNU General Public License version 2.\n\nCommercial license options are available also.\n\n# Contact\n\nQuestions? Comments? Suggestions? Get in touch!\n\n[bryan_gass@dfci.harvard.edu](mailto:bryan_gass@dfci.harvard.edu)\n\n<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/dfci_cornell_joint_logos.png width="750">\n',
    'author': 'Bryan Gass',
    'author_email': 'bryank123@live.com',
    'maintainer': 'Bryan Gass',
    'maintainer_email': 'bryank123@live.com',
    'url': 'https://github.com/Dana-Farber-AIOS/s4mer',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'python_requires': '>=3.8.0,<=3.8.13',
}


setup(**setup_kwargs)

