.. _api_ref:

.. currentmodule:: tobler

API reference
=============

.. _data_api:

Area Weighted
--------------
Area weighted approaches use the area of overlap between the source and target geometries to weight the
variables being assigned to the target

.. currentmodule:: tobler.area_weighted

.. autosummary::
   :toctree: generated/

    area_interpolate
    area_join


Dasymetric
-----------------------

Dasymetric approaches use auxiliary data in addition to use the area of overlap between the source and target geometries to weight the
variables being assigned to the target

.. currentmodule:: tobler.dasymetric

.. autosummary::
   :toctree: generated/

   extract_raster_features
   masked_area_interpolate


Model
---------
Model based approaches use additional spatial data, such as a land cover raster, to estimate the relationships between population
and the auxiliary data. It then uses that model to predict population levels at different scales

.. currentmodule:: tobler.model

.. autosummary::
   :toctree: generated/

   glm

Pycnophylactic
------------------
Pycnophylactic interpolation is based on `Tobler's technique <https://www.tandfonline.com/doi/abs/10.1080/01621459.1979.10481647>`_
for generating smooth, volume-preserving contour maps  

.. currentmodule:: tobler.pycno

.. autosummary::
   :toctree: generated/

   pycno_interpolate

Util
---------
Utility Functions

.. currentmodule:: tobler.util

.. autosummary::
   :toctree: generated/

   h3fy