
import geopandas as gpd
import pytest
from shapely.geometry import box

from tobler.util.dotdensity import _poly_to_dots, draw_points_by_column


@pytest.fixture
def simple_gdf():
    polygons = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    gdf = gpd.GeoDataFrame(
        {"pop_a": [10, 20, 15], "pop_b": [5, 8, 12], "label": ["x", "y", "z"]},
        geometry=polygons,
    )
    return gdf


def test_poly_to_dots_basic(simple_gdf):
    pts = _poly_to_dots(simple_gdf, category="pop_a", rng=42)
    assert isinstance(pts, gpd.GeoDataFrame)
    assert len(pts) == simple_gdf["pop_a"].sum()


def test_poly_to_dots_no_category(simple_gdf):
    with pytest.raises(ValueError, match="must pass a category"):
        _poly_to_dots(simple_gdf)


def test_poly_to_dots_missing_column(simple_gdf):
    with pytest.raises(ValueError, match="not in columns"):
        _poly_to_dots(simple_gdf, category="nonexistent")


def test_poly_to_dots_nonnumeric_column(simple_gdf):
    with pytest.raises(ValueError, match="must hold numeric data"):
        _poly_to_dots(simple_gdf, category="label")


def test_poly_to_dots_scale(simple_gdf):
    pts_full = _poly_to_dots(simple_gdf, category="pop_a", scale=1.0, rng=0)
    pts_half = _poly_to_dots(simple_gdf, category="pop_a", scale=0.5, rng=0)
    assert len(pts_half) < len(pts_full)


def test_poly_to_dots_has_poly_id(simple_gdf):
    pts = _poly_to_dots(simple_gdf, category="pop_a", rng=0)
    assert "poly_id" in pts.columns


def test_draw_points_basic(simple_gdf):
    result = draw_points_by_column(simple_gdf, columns=["pop_a", "pop_b"], rng=42)
    assert isinstance(result, gpd.GeoDataFrame)
    expected_len = simple_gdf["pop_a"].sum() + simple_gdf["pop_b"].sum()
    assert len(result) == expected_len


def test_draw_points_category_column(simple_gdf):
    result = draw_points_by_column(simple_gdf, columns=["pop_a", "pop_b"], rng=0)
    assert "category" in result.columns
    assert set(result["category"].unique()) == {"pop_a", "pop_b"}


def test_draw_points_category_dtype(simple_gdf):
    result = draw_points_by_column(simple_gdf, columns=["pop_a", "pop_b"], rng=0)
    assert result["category"].dtype.name == "category"


def test_draw_points_not_list_like(simple_gdf):
    with pytest.raises(ValueError, match="`categories` should be a list"):
        draw_points_by_column(simple_gdf, columns="pop_a")


def test_draw_points_scale(simple_gdf):
    result_full = draw_points_by_column(
        simple_gdf, columns=["pop_a"], scale=1.0, rng=0
    )
    result_scaled = draw_points_by_column(
        simple_gdf, columns=["pop_a"], scale=0.5, rng=0
    )
    assert len(result_scaled) < len(result_full)


def test_draw_points_reproducible(simple_gdf):
    r1 = draw_points_by_column(simple_gdf, columns=["pop_a"], rng=99)
    r2 = draw_points_by_column(simple_gdf, columns=["pop_a"], rng=99)
    assert r1.geometry.equals(r2.geometry)


def test_draw_points_geometry_within_polygons(simple_gdf):
    result = draw_points_by_column(simple_gdf, columns=["pop_a"], rng=0)
    union = simple_gdf.union_all()
    assert result.geometry.within(union).all()


def test_draw_points_single_column(simple_gdf):
    result = draw_points_by_column(simple_gdf, columns=["pop_b"], rng=0)
    assert len(result) == simple_gdf["pop_b"].sum()
    assert set(result["category"].unique()) == {"pop_b"}
