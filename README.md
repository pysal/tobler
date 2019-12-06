<p align="center">
<img src="docsrc/figs/tobler_long.png" height="200px">
</p>

[![Build Status](https://travis-ci.com/pysal/tobler.svg?branch=master)](https://travis-ci.org/pysal/tobler)
[![Coverage Status](https://coveralls.io/repos/github/pysal/tobler/badge.svg?branch=master)](https://coveralls.io/github/pysal/tobler?branch=master&service=github&kill_cache=1)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tobler)
![PyPI](https://img.shields.io/pypi/v/tobler)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/tobler)
![GitHub commits since latest release (branch)](https://img.shields.io/github/commits-since/pysal/tobler/latest)
[![DOI](https://zenodo.org/badge/202220824.svg)](https://zenodo.org/badge/latestdoi/202220824)

Tobler is a python package for areal interpolation, dasymetric mapping, and change of support.
<p></p>


`tobler` provides:

* areal interpolation for intensive and extensive variables	
* use raster files to improve interpolation accuracy
    + areal interpolation assuming data should only be allocated to specific land types
    + model-based interpolation using [regression](https://github.com/spatialucr/tobler/blob/master/examples/vectorized_raster_example.ipynb) approaches that incorporates auxiliary data
* Much more to come!


**Easily convert geospatial data from one polygonal representation to another:**

![](docs/figs/toy_census_tracts_example.png)

**Improve conversion accuracy by incorporating raster image data from satellites:**

![](docs/figs/raster_lattice_example.png)


## Installation

```bash
$ conda env create -f environment.yml
$ conda activate tobler 
$ python setup.py develop
```

Contribute
----------

PySAL-tobler is under active development and contributors are welcome.

If you have any suggestion, feature request, or bug report, please open a new [issue](https://github.com/pysal/tobler/issues) on GitHub. To submit patches, please follow the PySAL development [guidelines](http://pysal.readthedocs.io/en/latest/developers/index.html) and open a [pull request](https://github.com/pysal/tobler). Once your changes get merged, you’ll automatically be added to the [Contributors List](https://github.com/pysal/tobler/graphs/contributors).


License
-------

The project is licensed under the [BSD license](https://github.com/pysal/tobler/blob/master/LICENSE.txt).


Funding
-------

<img src="docs/figs/nsf_logo.jpg" width="50"> 

Award #1733705 [Neighborhoods in Space-Time Contexts](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1733705&HistoricalAwards=false)

 Award #1831615 [Scalable Geospatial Analytics for Social Science Research](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1831615)
