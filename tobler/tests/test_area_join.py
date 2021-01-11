import geopandas as gpd
import numpy as np
from shapely.geometry import Point

import pytest

from tobler.area_weighted import area_join


class TestAreaJoin:
    def setup_method(self):
        self.grid = gpd.points_from_xy(
            np.repeat(np.linspace(1, 10, 10), 10), np.tile(np.linspace(1, 10, 10), 10)
        ).buffer(0.5, cap_style=3)
        self.source = gpd.GeoDataFrame(
            {
                "floats": np.linspace(1, 10, 100),
                "ints": np.linspace(1, 100, 100, dtype="int"),
                "strings": np.array(["darribas", "is", "the", "king"] * 25),
            },
            geometry=self.grid,
        )

        self.target = gpd.GeoDataFrame(geometry=self.grid.translate(xoff=2.2, yoff=0.2))

    def test_area_join_float(self):
        result = area_join(self.source, self.target, "floats")
        assert (result.columns == ["geometry", "floats"]).all()
        np.testing.assert_almost_equal(result.floats.mean(), 6.409, 3)
        assert result.floats.dtype == float
        assert result.floats.isna().sum() == 20

    def test_area_join_ints(self):
        with pytest.warns(UserWarning, match="Cannot preserve dtype of"):
            result = area_join(self.source, self.target, "ints")

        assert (result.columns == ["geometry", "ints"]).all()
        np.testing.assert_almost_equal(result.ints.mean(), 60.5, 3)
        assert result.ints.dtype == object
        assert type(result.ints.iloc[0]) == int
        assert result.ints.isna().sum() == 20

    def test_area_join_strings(self):
        result = area_join(self.source, self.target, "strings")
        assert (result.columns == ["geometry", "strings"]).all()
        assert result.strings.dtype == object
        assert type(result.strings.iloc[0]) == str
        assert result.strings.isna().sum() == 20

    def test_area_join_array(self):
        with pytest.warns(UserWarning, match="Cannot preserve dtype of"):
            result = area_join(self.source, self.target, ["floats", "ints", "strings"])

        assert (result.columns == ["geometry", "floats", "ints", "strings"]).all()
        np.testing.assert_almost_equal(result.floats.mean(), 6.409, 3)
        assert result.floats.dtype == float
        assert result.floats.isna().sum() == 20
        np.testing.assert_almost_equal(result.ints.mean(), 60.5, 3)
        assert result.ints.dtype == object
        assert type(result.ints.iloc[0]) == int
        assert result.ints.isna().sum() == 20
        assert result.strings.dtype == object
        assert type(result.strings.iloc[0]) == str
        assert result.strings.isna().sum() == 20

    def test_area_join_error(self):
        target = self.target
        target["floats"] = 0
        with pytest.raises(ValueError, match="Column 'floats'"):
            area_join(self.source, target, "floats")
