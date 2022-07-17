import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

about = {}
with open("src/_version.py") as f:
    exec(f.read(), about)
version = about["__version__"]

setuptools.setup(
    name="s4mer",
    version=version,
    author="Bryan Gass, Jacob Rosenthal, Renato Umeton et al.",
    author_email="bryan_gass@dfci.harvard.edu",
    description="The test bench for how S4 works within various domains and regimes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    project_urls={
        "Documentation": "",  # TODO
        "Source Code": "https://github.com/Dana-Farber-AIOS/s4mer",
    },
    install_requires=[],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Framework :: Sphinx",
        "Framework :: Pytest",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
