# Installation

Currently, tobler supports Python >= [3.12]. Please make sure that you are operating in a Python 3 environment.

## Installing a released version

`tobler` is available on both conda and pip, and can be installed with any of

```bash
conda install -c conda-forge tobler
```

or

```bash
pixi install tobler
```

or

```bash
pip install tobler
```

## Installing a development from source

For working with a development version, we recommend [miniforge] or [pixi]. To get started, clone this repository or download it manually then `cd` into the directory and run the following commands:

**using conda**

```bash
conda env create -f environment.yml
conda activate tobler
pip install -e .
```

**using pixi**

*note*: as of this writing, pixi does not support relative paths (like "."), hence the expansion using the environment variable `$PWD`

```bash
pixi init --import environment.yml
pixi add --pypi --editable  "tobler @ file://$PWD"
```

You can also [fork] the [pysal/tobler] repo and create a local clone of your fork. By making changes to your local clone and submitting a pull request to [pysal/tobler], you can contribute to the tobler development.

[3.12]: https://docs.python.org/3.12/
[miniforge]: https://github.com/conda-forge/miniforge
[fork]: https://help.github.com/articles/fork-a-repo/
[pysal/tobler]: https://github.com/pysal/tobler
[python package index]: https://pypi.org/pysal/tobler/
[pixi]: https://pixi.prefix.dev/latest/
