import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, box

from tobler.util.dotdensity import _poly_to_dots, dot_density


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
    result = dot_density(simple_gdf, columns=["pop_a", "pop_b"], rng=42)
    assert isinstance(result, gpd.GeoDataFrame)
    expected_len = simple_gdf["pop_a"].sum() + simple_gdf["pop_b"].sum()
    assert len(result) == expected_len


def test_draw_points_category_column(simple_gdf):
    result = dot_density(simple_gdf, columns=["pop_a", "pop_b"], rng=0)
    assert "category" in result.columns
    assert set(result["category"].unique()) == {"pop_a", "pop_b"}


def test_draw_points_category_dtype(simple_gdf):
    result = dot_density(simple_gdf, columns=["pop_a", "pop_b"], rng=0)
    assert result["category"].dtype.name == "category"


def test_draw_points_not_list_like(simple_gdf):
    with pytest.raises(ValueError, match="`categories` should be a list"):
        dot_density(simple_gdf, columns="pop_a")


def test_draw_points_scale(simple_gdf):
    result_full = dot_density(simple_gdf, columns=["pop_a"], scale=1.0, rng=0)
    result_scaled = dot_density(simple_gdf, columns=["pop_a"], scale=0.5, rng=0)
    assert len(result_scaled) < len(result_full)


def test_draw_points_reproducible(simple_gdf):
    r1 = dot_density(simple_gdf, columns=["pop_a"], rng=99)
    r2 = dot_density(simple_gdf, columns=["pop_a"], rng=99)
    assert r1.geometry.equals(r2.geometry)


def test_draw_points_geometry_within_polygons(simple_gdf):
    result = dot_density(simple_gdf, columns=["pop_a"], rng=0)
    union = simple_gdf.union_all()
    assert result.geometry.within(union).all()


def test_draw_points_single_column(simple_gdf):
    result = dot_density(simple_gdf, columns=["pop_b"], rng=0)
    assert len(result) == simple_gdf["pop_b"].sum()
    assert set(result["category"].unique()) == {"pop_b"}


def test_draw_points_normal_cluster(simple_gdf):
    result = dot_density(simple_gdf, method="cluster_normal", columns=["pop_b"], rng=0)
    assert len(result) == simple_gdf["pop_b"].sum()
    assert set(result["category"].unique()) == {"pop_b"}


def test_draw_points_by_group_keeps_fractional_rows():
    # three rows of one group, each under half a point: rounding each row loses all
    gdf = gpd.GeoDataFrame(
        {"v": [0.3, 0.3, 0.4], "g": [1, 1, 1]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
    )
    assert len(dot_density(gdf, columns=["v"], rng=0)) == 0
    assert len(dot_density(gdf, columns=["v"], by="g", rng=0)) == 1


def test_draw_points_by_group_totals():
    gdf = gpd.GeoDataFrame(
        {"v": [2.4, 2.4, 1.2, 1.4], "g": ["a", "a", "b", "b"]},
        geometry=[box(i, 0, i + 1, 1) for i in range(4)],
    )
    pts = dot_density(gdf, columns=["v"], by="g", rng=0)
    per_group = gdf["g"].to_numpy()[pts["poly_id"].to_numpy()]
    assert (per_group == "a").sum() == 5
    assert (per_group == "b").sum() == 3


def test_draw_points_by_missing_column(simple_gdf):
    with pytest.raises(ValueError, match="not in columns"):
        dot_density(simple_gdf, columns=["pop_a"], by="nonexistent")


def test_draw_points_int_seed_draws_independently():
    # identical geometries and two categories must not repeat the same points
    gdf = gpd.GeoDataFrame({"a": [5, 5], "b": [5, 5]}, geometry=[box(0, 0, 10, 10)] * 2)
    pts = dot_density(gdf, columns=["a", "b"], rng=0)
    xy = {(round(p.x, 9), round(p.y, 9)) for p in pts.geometry}
    assert len(xy) == len(pts) == 20


def test_draw_points_each_point_in_its_own_polygon(simple_gdf):
    pts = dot_density(simple_gdf, columns=["pop_a", "pop_b"], rng=0)
    own = simple_gdf.geometry.loc[pts["poly_id"]].to_numpy()
    assert shapely.contains(own, pts.geometry.to_numpy()).all()


def test_draw_points_uniform_in_multipolygon():
    # two equal parts far apart: points should split about evenly between them
    gdf = gpd.GeoDataFrame(
        {"v": [10_000]}, geometry=[box(0, 0, 1, 1).union(box(100, 0, 101, 1))]
    )
    pts = dot_density(gdf, columns=["v"], rng=0)
    assert len(pts) == 10_000
    assert (pts.geometry.x < 50).mean() == pytest.approx(0.5, abs=0.02)


def test_draw_points_zero_area_polygon_warns():
    gdf = gpd.GeoDataFrame(
        {"v": [3, 3]},
        geometry=[box(0, 0, 1, 1), shapely.Polygon([(0, 0), (1, 1), (2, 2)])],
    )
    with pytest.warns(UserWarning, match="zero area"):
        pts = dot_density(gdf, columns=["v"], rng=0)
    assert len(pts) == 3


def test_draw_points_lines_fall_back_to_geopandas():
    gdf = gpd.GeoDataFrame(
        {"v": [4, 5]}, geometry=[box(0, 0, 1, 1), LineString([(5, 0), (6, 0)])]
    )
    pts = dot_density(gdf, columns=["v"], rng=0)
    assert len(pts) == 9
    on_line = pts[pts["poly_id"] == 1].geometry
    assert len(on_line) == 5
    assert np.allclose(on_line.y, 0)


def test_draw_points_leaves_geometries_unprepared(simple_gdf):
    dot_density(simple_gdf, columns=["pop_a"], rng=0)
    assert not shapely.is_prepared(simple_gdf.geometry.to_numpy()).any()


def test_draw_points_poisson_cluster(simple_gdf):
    result = dot_density(simple_gdf, method="cluster_poisson", columns=["pop_b"], rng=0)
    assert len(result) == simple_gdf["pop_b"].sum()
    assert set(result["category"].unique()) == {"pop_b"}
