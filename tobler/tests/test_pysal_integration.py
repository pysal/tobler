"""lightweight test for pysal metapckage that functions import."""

def test_imports():
    import quilt3
    from tobler.dasymetric import glm, glm_pixel_adjusted, masked_area_interpolate
    from tobler.area_weighted import area_interpolate
    from tobler.data import store_rasters
