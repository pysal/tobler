from tobler.data import store_rasters
import os

def test_raster_download():
    store_rasters(os.getcwd())
