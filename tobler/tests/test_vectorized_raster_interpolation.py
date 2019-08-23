from tobler.vectorized_raster_interpolation import *
from tobler.data import *

import rasterio
import unittest
import libpysal

import geopandas as gpd
import numpy as np

from shapely.geometry import Polygon
from shapely.ops import cascaded_union


class VectorizedRasterInterpolation_Tester(unittest.TestCase):
    def test_VectorizedRasterInterpolation(self):
        local_raster_path = fetch_quilt_path('nlcd_2011')
        nlcd_raster = rasterio.open(local_raster_path)
        
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'TOT_POP']]
        df.crs = {'init': 'epsg:4326'}
        
        # Shrinking the spatial extent
        df = df[((df.centroid.x < -121.4) & (df.centroid.x > -121.5) & (df.centroid.y > 38.5) & (df.centroid.y < 38.8))]
        df = df.set_geometry('geometry')
        
        weights = return_weights_from_regression(df, local_raster_path, 'TOT_POP')
        
        correspondence_table = create_non_zero_population_by_pixels_locations(df, nlcd_raster, 'TOT_POP', weights)
        
        boundary = gpd.GeoSeries(cascaded_union(df.geometry)).buffer(0.001) # The 'buffer' method prevents some unusual points inside the state
        df_region_pre = gpd.GeoDataFrame(gpd.GeoSeries(boundary))
        df_region = df_region_pre.rename(columns={0:'geometry'}).set_geometry('geometry')
        
        low_left  = -121.6
        low_right = -121.3
        up_left = 38.4
        up_right = 39
        
        thickness = 15
        
        # This snippet was inspired in https://github.com/pysal/libpysal/blob/aa7882e7877b962f4269ea86a612dfc58152e5c6/libpysal/weights/user.py#L95
        aux = list()
        for i in np.linspace(low_left, low_right, num = thickness):
                for j in np.linspace(up_left, up_right, num = thickness):
                    
                    # Each width 'jump' must be at the same order of the grid constructed
                    ll = i, j
                    ul = i, j + np.diff(np.linspace(up_left, up_right, num = thickness))[0]
                    ur = i + np.diff(np.linspace(low_left, low_right, num = thickness))[0], j + np.diff(np.linspace(up_left, up_right, num = thickness))[0]
                    lr = i + np.diff(np.linspace(low_left, low_right, num = thickness))[0], j
                    aux.append(Polygon([ll, ul, ur, lr, ll]))
                    
        polys2 = gpd.GeoSeries(aux)
        
        envgdf = gpd.GeoDataFrame(polys2)
        envgdf_final = envgdf.rename(columns={0:'geometry'}).set_geometry('geometry')
        
        res_union = gpd.overlay(df_region, envgdf_final, how='intersection')
        res_union.crs = df.crs
        
        interpolated = calculate_interpolated_population_from_correspondence_table(res_union, nlcd_raster, correspondence_table)
        
        x = np.array(interpolated['interpolated_population'])
        x.sort()
        
        np.testing.assert_almost_equal(x, np.array([2.29409707e+01, 6.48706970e+01, 1.21688696e+03, 1.83378287e+03,
       1.87042079e+03, 1.90379325e+03, 2.25049982e+03, 3.21650854e+03,
       3.41236645e+03, 3.71181098e+03, 5.12532031e+03, 5.75600935e+03,
       8.63080789e+03, 9.26963356e+03, 1.00591807e+04, 1.04378577e+04,
       1.29541218e+04, 1.56270494e+04, 1.89447344e+04, 1.91294133e+04,
       1.93945973e+04, 2.06785383e+04, 2.15440941e+04, 2.29075806e+04,
       2.30862412e+04, 2.49861096e+04, 2.97869188e+04, 2.99332151e+04,
       3.12133550e+04, 3.18697373e+04, 3.19341907e+04, 3.37013110e+04,
       4.10443313e+04, 4.49694860e+04]), decimal = 3)

if __name__ == '__main__':
    unittest.main()
