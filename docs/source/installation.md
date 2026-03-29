# Installation

Currently, tobler supports Python >= [3.12]. Please make sure that you are operating in a Python 3 environment.

## Installing a released version

`tobler` is available on both conda and pip, and can be installed with either

```bash
conda install -c conda-forge tobler
```

or

```bash
pip install tobler
```

## Installing a development from source

For working with a development version, we recommend [miniforge]. To get started, clone this repository or download it manually then `cd` into the directory and run the following commands:

```bash
conda env create -f environment.yml
conda activate tobler
pip install -e .
```

You can also [fork] the [pysal/tobler] repo and create a local clone of your fork. By making changes to your local clone and submitting a pull request to [pysal/tobler], you can contribute to the tobler development.

[3.12]: https://docs.python.org/3.12/
[miniforge]: https://github.com/conda-forge/miniforge
[fork]: https://help.github.com/articles/fork-a-repo/
[pysal/tobler]: https://github.com/pysal/tobler
[python package index]: https://pypi.org/pysal/tobler/
