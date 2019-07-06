# tobler: a library for spatial interpolation in Python

**Easily perform area interpolation for different set of set of polygons:**

![](figs/toy_census_tracts_example.png)

**Perform enhanced interpolation using raster image files from satellites:**

![](figs/raster_lattice_example.png)

## Functionalities

In `tobler` you can execute:

* areal interpolation for intensive and extensive variables
	+ overlay
	+ binning
	
* use raster files to improve interpolation of variables'
    + areal interpolation assuming only specific land types
    + [regression using vectorized version](https://github.com/spatialucr/tobler/blob/master/examples/vectorized_raster_example.ipynb)
    + regression leveraged by scanlines to perform interpolation

* [harmonize different set of unmatching polygons with different methods](https://github.com/spatialucr/tobler/blob/master/examples/harmonizing_community_example.ipynb)

## Installation

```bash
$ conda env create -f environment.yml
$ conda activate tobler 
$ python setup.py develop
```

## Roadmap

* TODO r-tree or binning for indexing and table generation
* TODO allow for weights parameter
* TODO hybrid harmonization
* TODO union harmonization
* TODO nlcd auxiliary regressions