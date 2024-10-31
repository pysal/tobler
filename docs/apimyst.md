# API 

This is a beta version of the documentation written in [MyST](https://myst-parser.readthedocs.io/en/latest/index.html) with [autodoc2](https://sphinx-autodoc2.readthedocs.io/en/latest/quickstart.html)

## Area Weighted Interpolation

Area weighted approaches use the area of overlap between the source and target geometries to weight the variables being assigned to the target.

```{autodoc2-summary}  tobler.area_weighted.area_interpolate.area_interpolate
```

```{autodoc2-summary}  tobler.area_weighted.area_join.area_join
```

## Dasymetric Interpolation

Dasymetric approaches use auxiliary data in addition to use the area of overlap
between the source and target geometries to weight the variables being assigned
to the target

```{autodoc2-summary}  tobler.dasymetric.raster_tools.extract_raster_features
```

```{autodoc2-summary}  tobler.dasymetric.masked_area_interpolate.masked_area_interpolate
```

## Pycnophylactic Interpolation

Pycnophylactic interpolation is based on
[Tobler's technique](https://www.tandfonline.com/doi/abs/10.1080/01621459.1979.10481647)
for generating smooth, volume-preserving contour maps

```{autodoc2-summary}  tobler.pycno.pycno.pycno_interpolate
```

## Utility Functions

```{autodoc2-summary}  tobler.util.util.h3fy
```

```{autodoc2-summary}  tobler.util.util.circumradius
```