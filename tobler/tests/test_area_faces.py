import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

import pytest

from tobler.area_weighted import area_faces, area_buffer


class TestAreaFaces:
    def setup_method(self):
        polys1 = gpd.GeoSeries([Polygon([(0,0), (10,0), (10,5), (0,5)]),
                            Polygon([(0,5), (0,10),  (5,10), (5,5)]),
                            Polygon([(5,5), (5,10), (7,10), (7,5)]),
                            Polygon([(7,5), (7,10), (10,10), (10,5)]) ]
                            )


        buffer = gpd.GeoSeries([Polygon([ (0,0), (0, 10), (6, 10), (6,0)])])


        df1 = gpd.GeoDataFrame({'geometry': polys1})
        df2 = gpd.GeoDataFrame({'geometry': buffer})
        df1['population'] = [ 500,  200, 100, 50]
        df1['pci'] = [75, 100, 40, 30]
        df1['income'] = df1['population'] * df1['pci']
        df2['population'] = 10000
        df2['pci'] = 80
        self.source = df1
        self.target = df2


    def test_area_faces(self):
        result = area_faces(self.source, self.target,
                            extensive_variables=['population']) 
        assert (result.shape == (6,2))
        pop_values = np.array([299.99998212, 200., 50., 199.99998808, 50.  , 50.  ])
        np.testing.assert_almost_equal(result.population.values, pop_values, 2)

    def test_area_faces_2_1(self):
        result = area_faces(self.target, self.source,
                            extensive_variables=['population'],
                            intensive_variables=['pci']) 
        assert (result.shape == (6,3))
        pop_values = np.array([5000, 4166.67, 833.33, 0, 0, 0 ])
        np.testing.assert_almost_equal(result.population.values, pop_values, 2)
        pci_values = np.array([80, 80, 80, 0, 0, 0 ])
        np.testing.assert_almost_equal(result.pci.values, pci_values, 2)

    def test_area_buffer(self):
        result = area_buffer(self.source, self.target)
        preds = ['partial', 'within', 'partial', 'disjoint']
        assert (result.right_relation.tolist() == preds)

