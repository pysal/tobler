.. Installation

.. highlight:: rst

.. role:: python(code)
    :language: python


Installation
===============

tobler supports Python `3.6`_ and `3.7`_ only. Please make sure that you are
operating in a Python 3 environment.

Installing a released version
------------------------------
``tobler`` is available on both conda and pip, and can be installed with either

.. code-block:: bash

    conda install -c conda-forge tobler

or

.. code-block:: bash

    pip install tobler


Installing a development from source
-------------------------------------
For working with a development version, we recommend `anaconda`_. To get started, clone this repository or download it manually then ``cd`` into the directory and run the following commands:

.. code-block:: bash

    conda env create -f environment.yml
    source activate tobler 
    python setup.py develop

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
