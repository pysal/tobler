import unittest
import numpy as np
import geopandas as gpd
from tobler.area_weighted import (area_tables,
                                  area_tables_binning,
                                  area_interpolate,
                                  area_interpolate_binning)

from shapely.geometry import Polygon


class AreaTablesAreaInterpolate_Tester(unittest.TestCase):
    def test_AreaTablesAreaInterpolate(self):
        polys1 = gpd.GeoSeries([Polygon([(0,0), (10,0), (10,5), (0,5)]),
                                Polygon([(0,5), (0,10),  (10,10), (10,5)])])
        
        polys2 = gpd.GeoSeries([Polygon([(0,0), (5,0), (5,7), (0,7)]),
                                Polygon([(5,0), (5,10),  (10,10), (10,0)]),
                                Polygon([(0,7), (0,10), (5,10), (5,7)  ])
                                ])
        
        df1 = gpd.GeoDataFrame({'geometry': polys1})
        df2 = gpd.GeoDataFrame({'geometry': polys2})
        df1['population'] = [ 500,  200]
        df1['pci'] = [75, 100]
        df1['income'] = df1['population'] * df1['pci']
        
        
        res_union = gpd.overlay(df1, df2, how='union')
        
        result_area = area_tables(df1, res_union)
        result_area_binning = area_tables_binning(df1, res_union)
        
        np.testing.assert_almost_equal(result_area[0], np.array([[25., 25.,  0.,  0.,  0.],
       [ 0.,  0., 10., 15., 25.]]), decimal = 3)
    
        np.testing.assert_almost_equal(result_area[1], np.array([[1., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 1., 0.]]), decimal = 3)
        
        np.testing.assert_almost_equal(result_area_binning.toarray(), np.array([[25.,  0., 25.,  0.,  0.], 
                                       [ 0., 10.,  0., 25., 15.]]), decimal = 3)
    
        result_inte = area_interpolate(df1, res_union, extensive_variables = ['population', 'income'], intensive_variables = ['pci'])
        result_inte_binning = area_interpolate_binning(df1, res_union, extensive_variables = ['population', 'income'], intensive_variables = ['pci'])
        
        
        np.testing.assert_almost_equal(result_inte[0], np.array([[  250., 18750.],
       [   40.,  4000.],
       [  250., 18750.],
       [  100., 10000.],
       [   60.,  6000.]]), decimal = 3)
    
    
        np.testing.assert_almost_equal(result_inte[1], np.array([[ 75.],
       [100.],
       [ 75.],
       [100.],
       [100.]]), decimal = 3)       
        
        np.testing.assert_almost_equal(result_inte_binning[0], np.array([[  250.        , 18750.        ],
           [   39.99999762,  3999.99976158],
           [  250.        , 18750.        ],
           [  100.        , 10000.        ],
           [   59.99999642,  5999.99964237]]), decimal = 3)
    
        np.testing.assert_almost_equal(result_inte_binning[1], np.array([[ 75.], [100.], [ 75.], [100.], [100.]]), decimal = 3)

if __name__ == '__main__':
    unittest.main()
