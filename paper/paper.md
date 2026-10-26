---
title: 'tobler: Spatial interpolation and areal analysis for Python'
tags:
- Python
- geographic information science
- interpolation
- spatial analysis
date: "25 March 2023"
output: pdf_document
authors:
- name: Elijah Knaap
  orcid: "0000-0001-7520-2238"
  equal-contrib: true
  affiliation: 1
- name: Sergio J. Rey
  orcid: "0000-0001-5857-9762"
  equal-contrib: true
  affiliation: 2
- name: Renan X. Cortes
  orcid: "0000-0002-1889-5282"
  equal-contrib: true
  affiliation: 3
- name: Martin Fleischmann
  orcid: "0000-0003-3319-3366"
  equal-contrib: true
  affiliation: 4
- name: James D. Gaboardi
  orcid: "0000-0002-4776-6826"
  equal-contrib: true
  affiliation: 5
- name: Daniel Arribas-Bel
  orcid: "0000-0002-6274-1619"
  equal-contrib: true
  affiliation: 6
bibliography: paper.bib
affiliations:
- name: University of California, Irvine
  index: 1
- name: San Diego State University
  index: 2
- name: Federal University of Rio Grande do Sul
  index: 3
- name: Charles University, Faculty of Science
  index: 4
- name: Geospatial Science and Human Security, Oak Ridge National Laboratory
  index: 5
- name: University of Liverpool
  index: 6
---

<!--

NEW JOSS REQUIREMENTS

Length

* longer paper acceptable (but not necessarily encouraged...)
*  750-1750

Sections

https://joss.readthedocs.io/en/latest/paper.html#:~:text=Your%20paper%20must-,include,-the%20following%20required

* Summary
* Statement of need
* State of the field
* Software design
* Research impact statement
* AI usage disclosure

-->


# Summary


`tobler` is an open-source Python library for spatial interpolation and areal data transformation, designed to support a wide range of workflows in geographic information science (GIScience), spatial data science, and applied spatial analysis. The package provides a unified, extensible framework for transferring data between incompatible zonal systems—an essential task when working with spatially aggregated data such as census geographies, administrative boundaries, or service areas.

Areal interpolation is a foundational problem in spatial analysis, arising whenever data collected over one set of polygons (source zones) must be reallocated to another (target zones) with different spatial configurations. `tobler` implements a suite of methods for this task, ranging from simple area-weighted interpolation to more advanced dasymetric and model-based approaches. The package is designed to operate natively on GeoPandas GeoDataFrames, enabling seamless integration into modern Python-based geospatial workflows.

As part of the Python Spatial Analysis Library (PySAL) ecosystem [@pysal2007; @rey2022pysalecosystem], `tobler` emphasizes composability, transparency, and reproducibility while providing robust implementations of areal interpolation methods for both applied and methodological research.

# Statement of need

Spatial data are commonly aggregated to administrative units such as census tracts, ZIP codes, or political boundaries, which often differ across datasets or change over time. As a result, transferring variables between incompatible zoning systems is a common challenge for longitudinal analysis, data integration, and cross-scale comparison, particularly in the social and behavioral sciences. While Python packages such as `PyKrige` [@benjamin_murphy_2025_17372225] and `pyinterpolate` [@molinski2022PyinterpolateSpatial] support *continuous* spatial interpolation, they are not designed for interpolation between discrete zoning systems.

Traditional GIS software (e.g., ArcGIS, QGIS) provides tools for areal interpolation, but these implementations are often embedded in graphical interfaces, limiting reproducibility and automation. Moreover, they may not expose the full range of methodological options or allow for easy integration into data science pipelines.

Prior to `tobler`, the Python ecosystem support for areal interpolation was fragmented or limited. While foundational libraries such as GeoPandas provide data structures and geometric operations, they do not natively implement interpolation methods. As a result, users often relied on *ad hoc* scripts or external tools, leading to inconsistent workflows and potential methodological errors.

`tobler` addresses these challenges by providing:

* A coherent and well-documented API for areal interpolation
* Native integration with GeoPandas and the PySAL ecosystem
* Area-weighted, dasymetric, and model-based interpolation methods
* Reproducible workflows

These capabilities make `tobler` particularly valuable for researchers and practitioners in fields such as geography, urban planning, public health, environmental science, and regional economics, where spatial data integration is a routine requirement.

# State of the field

`tobler` is a component of the PySAL ecosystem [@pysal-2026], which provides a comprehensive suite of tools for spatial analysis in Python. This ecosystem consists of four layers of related packages:

* `libpysal` for foundational spatial data structures and utilities [@libpysal-2026]
* `explore` for exploratory spatial analysis
* `model` for spatial modeling
* `viz` provides classification schemes for choropleth mapping

Within this ecosystem, **`tobler`** occupies a critical niche by providing the data integration and interpolation methods necessary to harmonize datasets before analysis in the **explore** or **model** phases.

`tobler` complements these packages by addressing the specific problem of spatial data transformation between incompatible zonal systems. Compared to desktop GIS platforms, `tobler` offers advantages like **Reproducibility**, **Transparency**, **Extensibility**, and **Integration**.

While similar functionality exists in other ecosystems (e.g., R packages such as `areal` [@Preneretal2019] or `sf`-based workflows [@RJ-2018-009]), `tobler` provides a native solution for Python users.

# Software design

`tobler` is designed with attention to both computational efficiency and usability. Spatial overlay operations, which are central to areal interpolation, can be computationally intensive for large datasets. The package leverages vectorized operations and efficient geometric libraries (via GeoPandas [@geopandas-2026] and Shapely [@shapely-2026]) to handle these tasks.

The API design emphasizes clarity and consistency, with function signatures that explicitly distinguish between extensive and intensive variables. Intensive variables are things like rates or percentages that must be averaged when aggregating, whereas extensive variables are things like counts that must be summed during aggregation.

Additionally, `tobler` is developed with testing and documentation standards consistent with the Scientific Python ecosystem, ensuring reliability and maintainability.

## Core Functionality

`tobler` organizes its functionality around several key interpolation paradigms, each corresponding to different assumptions about how variables are distributed within source zones.

### Area-weighted interpolation

Area-weighted interpolation assumes that variables are uniformly distributed within each source zone and allocates values to target zones in proportion to the area of overlap.

`tobler` provides efficient implementations for both **extensive variables** (e.g., population counts) and **intensive variables** (e.g., rates or densities), ensuring appropriate handling of each type [@goodchild1980areal]. The library also supports pycnophylactic adjustments to preserve totals where required [@tobler1979SmoothPycnophylactic].

### Dasymetric interpolation

Dasymetric interpolation uses ancillary data (e.g., land use or land cover) to model within-zone heterogeneity [@mennis2006IntelligentDasymetric; @Eicher2001dasy; @Reibel2007]. `tobler` supports both vector- and raster-based dasymetric workflows, allowing users to integrate a wide range of auxiliary datasets. This is particularly useful in urban and environmental applications where fine-scale heterogeneity is important.

### Model-based interpolation

Beyond deterministic approaches, `tobler` includes model-based methods that use statistical or machine learning techniques to estimate spatial distributions. These approaches incorporate covariates and capture more complex spatial patterns, providing improved accuracy in many contexts [@flowerdew1992DevelopmentsAreal; @flowerdewMethodFittingGravity1982].

## Integration with GeoPandas

All interpolation methods in `tobler` operate directly on GeoPandas GeoDataFrames and return GeoDataFrames, enabling seamless integration with the broader Python geospatial ecosystem.

## Example workflow

A typical area-weighted interpolation in `tobler` is implemented as follows:

```python
from tobler.area_weighted import area_interpolate

result = area_interpolate(
    source_df,
    target_df,
    extensive_variables=["population"],
    intensive_variables=["income"]
)
```

This operation transfers population counts and income measures from the source geometries to the target geometries, handling each variable type (extensive/intensive) appropriately.

When additional information about within-zone heterogeneity is available, dasymetric interpolation can be used to refine estimates. For example, population counts may be redistributed using a land cover raster to exclude uninhabited areas:

```python
from tobler.dasymetric import masked_area_interpolate

result = masked_area_interpolate(
    raster="raster_file_name.tif",
    source_df,
    target_df,
    pixel_values = [21,22,23,24],
    extensive_variables=["population"]
)
```

Similarly, the user can execute a model-based approach using the `tobler.model.glm` function.

\autoref{fig:emp_male_maps} illustrates an example comparing interpolated values derived from different spatial configurations, highlighting how results may vary depending on the underlying geometry and interpolation approach.

![Example of `tobler` usage for a single extensive variable -- employed male population -- (no intensive variable) in Charleston, SC, comparing census tracts and ZCTAs.\label{fig:emp_male_maps}](figs/emp_male_maps.png)

# Research impact statement

The package is actively used by the research community to transfer the data between various types of geographic boundaries. This is not limited to specific applications but covers use cases from continental analysis of emissions and health [@laporta2024Urban], analysis of urban form and function [@fleischmann2022Geographical], redistribution of census data to school districts for assessment of the Clean School Bus Rebate Program [@osia2025Infrastructure], quantification of radon exposure [@lee2026QuantifyingMean], and harmonization of vector and raster data for computer vision tasks [@fleischmann2024Decoding].

Moreover, the package is relied on in downstream software such as `atlasbr` for harmonization of Brazilian urban data [@oliveira_paiva_neto_atlasbr], and is referred to in the `pygridmap` package by Eurostat [@grazzini_gaffuri_pygridmap] as a reference implementation.
The `tobler` package has made tangible contributions to spatial science, pedagogy, and applications in government and industry. In academia, the package is used as part of a data-processing pipeline for research that examines the spatial-contextual influence on a variety of outcomes, including segregation [@wei2022ReducingRacial], housing policy [@rey2022LegacyRedlining], education policy [@rey2024MeasuringSpatial; @osia2025Infrastructure], and pollution exposure [@lee2026QuantifyingMean; @laporta2024Urban]. It is also used in environmental science [@hu2023MethodologicalChallenges] and regionalization research [@feng2022MaxpcompactregionsProblem].

In spatial data science education, `tobler` has become an integral part of  many curricula. It is included in popular pedagogical resources, including two textbooks [@reyGeographicDataScience2023; @knaapUrbanAnalysis2026], and is taught in graduate and undergraduate courses in universities across the globe, including the University of California (Berkeley, Irvine, Riverside, and Santa Barbara campuses), San Diego State University, Charles University, University of Liverpool,  Northern Arizona University, and Temple University, among others.

In the public sector, the `tobler` package is used as part of a processing pipeline that powers urban planning and policymaking, including the  [DemoLand](https://www.turing.ac.uk/research/research-projects/demoland) project from the Alan Turing Institute, developed as a part of the British National Land Data Programme [@geospatialcommission2023]. The project piloted predictive modelling based on data interpolated to a unified hexagonal grid and is scaling the approach nation-wide.

# AI usage disclosure

No generative AI or LLMs were used for code production for `tobler` or the writing of this paper.

# Acknowledgements

`tobler` is developed by the PySAL community and builds on decades of research in areal interpolation and spatial data science.

Funding from National Science Foundation Grants [2345820](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2345820) and
[1831615](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=1831615&HistoricalAwards=false) have supported `tobler` development.

The following acknowledgement applies to James D. Gaboardi:

> This manuscript has been authored in part by UT-Battelle LLC under contract DE-AC05-00OR22725 with the US Department of Energy (DOE). The US government retains and the publisher, by accepting the article for publication, acknowledges that the US government retains a nonexclusive, paid-up, irrevocable worldwide license to publish or reproduce the published form of this manuscript, or allow others to do so, for US government purposes. DOE will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).

The following acknowledgement applies to Daniel Arribas-Bel and Martin
Fleischmann:

> Funding is acknowledged from UK Research and Innovation (UKRI) through the Economic and Social Research Council's grant “Learning an urban grammar from satellite data through AI”, project reference (ES/ T005238/1).

# References
