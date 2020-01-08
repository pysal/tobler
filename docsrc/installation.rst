.. Installation

.. highlight:: rst

.. role:: python(code)
    :language: python


Installation
===============

tobler supports Python `3.6`_ and `3.7`_ only. Please make sure that you are
operating in a Python 3 environment.

Installing released version
---------------------------

You can install tobler with :python:`pip install tobler`

Installing development version
------------------------------
The recommended method for installing tobler is with `anaconda`_. To get started with the development version, clone this repository or download it manually then ``cd`` into the directory and run the following commands::

$ conda env create -f environment.yml
$ source activate tobler 
$ python setup.py develop

You can  also `fork`_ the `pysal/tobler`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/tobler`_, you can
contribute to the tobler development.

.. _3.6: https://docs.python.org/3.6/
.. _3.7: https://docs.python.org/3.7/
.. _Python Package Index: https://pypi.org/pysal/tobler/
.. _pysal/tobler: https://github.com/pysal/tobler
.. _fork: https://help.github.com/articles/fork-a-repo/
.. _anaconda: https://www.anaconda.com/download/ 
