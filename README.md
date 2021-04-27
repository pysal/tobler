<p align="center">
<img src="docs/figs/tobler_long.png" height="200px">
</p>

![CI Tests](https://github.com/pysal/tobler/workflows/Unit%20Tests/badge.svg)
[![codecov](https://codecov.io/gh/pysal/tobler/branch/master/graph/badge.svg?token=XO4SilfBEb)](https://codecov.io/gh/pysal/tobler)![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tobler)
![PyPI](https://img.shields.io/pypi/v/tobler)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/tobler)
![GitHub commits since latest release (branch)](https://img.shields.io/github/commits-since/pysal/tobler/latest)
[![DOI](https://zenodo.org/badge/202220824.svg)](https://zenodo.org/badge/latestdoi/202220824)

# PySAL `tobler`

`tobler` is a python package for areal interpolation, dasymetric mapping, change of support, and small area estimation. It provides a suite of tools with a simple interface for transferring data from one polygonal representation to another. Common examples include standardizing census data from different time periods to a single representation (i.e. to overcome boundary changes in successive years), or the conversion of data collected at different spatial scales into shared units of analysis (e.g. converting zip code and neighborhood data into a regular grid). `tobler` is part of the [PySAL](https://pysal.org) family of packages for spatial data science and provides highly performant implementations of basic and advanced interpolation methods, leveraging [`pygeos`](https://pygeos.readthedocs.io/en/latest/) to optimize for multicore architecture. The package name is an homage to the legendary quantitative geographer [Waldo Tobler](https://en.wikipedia.org/wiki/Waldo_R._Tobler), a pioneer in geographic interpolation methods, spatial analysis, and computational social science.

![DC tracts to hexgrid](docs/_static/images/notebooks_census_to_hexgrid_25_1.png)

## Interpolation Methods

`tobler` provides functionality for three families of spatial interpolation methods. The utility of each technique depends on the context of the problem and varies according to e.g. data availability, the properties of the interpolated variable, and the resolution of source and target geometries. For a further explanation of different interpolation techniques, please explore some of the field's [background literature](https://pysal.org/tobler/references.html)

### Area Weighted

Areal interpolation uses the area of overlapping geometries to apportion variables. This is the simplest method with no additional data requirements, aside from input and output geometries, however this method is also most susceptible to the [modifiable areal unit problem](https://en.wikipedia.org/wiki/Modifiable_areal_unit_problem).

### Dasymetric

Dasymetric interpolation uses auxiliary data to improve estimation, for example
by constraining the areal interpolation to areas that are known to be inhabited. Formally, `tobler` adopts a binary dasymetric approach, using auxiliary data to define which land is available or unavailable for interpolation. The package can incorporate additional sources such as

* raster data such as satellite imagery that define land types
* vector features such as roads or water bodies that define habitable or inhabitable land

either (or both) of which may be used to help ensure that variables from the source geometries are not allocated to inappropriate areas of the target geometries. Naturally, dasymetric approaches are sensitive to the quality of ancillary data and underlying assumptions used to guide the estimation.

### Model-based

Model-based interpolation uses [spatial] statistical models to estimate a relationship between the target variable and a set of covariates such as physical features, administrative designations, or demographic and architectural characteristics. Model-based approaches offer the ability to incorporate the richest set of additional data, but they can also difficult to wield in practice because the true relationship between variables is never known. By definition, some formal assumptions of regression models are violated because the target variable is always predicted using data from a different spatial scale than it was estimated.

### Extensions

`tobler` is under active development and will continue to incorporate emerging interpolation methods as they are introduced to the field. We welcome any and all contributions and if you would like to propose an additional method for adoption please raise an issue for discussion or open a new pull request!

![Charleston zipcodes to tracts](docs/_static/images/tobler3.png)

![raster example](docs/figs/raster_lattice_example.png)

## Installation

```bash
$ conda env create -f environment.yml
$ conda activate tobler 
$ python setup.py develop
```

## Contribute

PySAL-tobler is under active development and contributors are welcome.

If you have any suggestion, feature request, or bug report, please open a new [issue](https://github.com/pysal/tobler/issues) on GitHub. To submit patches, please follow the PySAL development [guidelines](http://pysal.readthedocs.io/en/latest/developers/index.html) and open a [pull request](https://github.com/pysal/tobler). Once your changes get merged, you’ll automatically be added to the [Contributors List](https://github.com/pysal/tobler/graphs/contributors).

## License

The project is licensed under the [BSD license](https://github.com/pysal/tobler/blob/master/LICENSE.txt).

## Funding

<img src="docs/figs/nsf_logo.jpg" width="50"> 

Award #1733705 [Neighborhoods in Space-Time Contexts](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1733705&HistoricalAwards=false)

Award #1831615 [Scalable Geospatial Analytics for Social Science Research](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1831615)
