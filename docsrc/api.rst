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
    area_tables
    area_tables_binning
    area_tables_raster


Dasymetric
-----------------------

Dasymetric approaches use auxiliary data in addition to use the area of overlap between the source and target geometries to weight the 
variables being assigned to the target

.. currentmodule:: tobler.dasymetric

.. autosummary::
   :toctree: generated/

   glm
   glm_pixel_adjusted
   masked_area_interpolate
   

Data
-----------------------

.. currentmodule:: tobler.data

.. autosummary::
   :toctree: generated/

    store_rasters