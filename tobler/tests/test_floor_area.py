"""Tests for floor-area dasymetric interpolation and dot density."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from tobler.dasymetric import (
    floor_area_dot_density,
    floor_area_interpolate,
    stack_footprints,
)
from tobler.dasymetric.vector_tools import _SOURCE, _floor_rows

CRS = 3857


@pytest.fixture
def one_tract():
    """One tract; A is 1 floor and B is 9 floors, with equal footprints."""
    tract = gpd.GeoDataFrame(
        {"pop": [1000.0]}, geometry=[box(0, 0, 100, 100)], index=["t"], crs=CRS
    )
    bldgs = gpd.GeoDataFrame(
        {"height": [3.0, 27.0]},
        geometry=[box(0, 0, 10, 10), box(80, 0, 90, 10)],
        index=["A", "B"],
        crs=CRS,
    )
    targets = gpd.GeoDataFrame(geometry=list(bldgs.geometry), index=["A", "B"], crs=CRS)
    return tract, bldgs, targets


@pytest.fixture
def two_tracts():
    """Two tracts split at x=50; building C (3 floors) straddles the boundary.

    Floor area: w = A 100 + half of C 100 * 3 = 400; e = B 900 + 300 = 1200.
    """
    tracts = gpd.GeoDataFrame(
        {"pop": [1000.0, 600.0], "rate": [10.0, 20.0]},
        geometry=[box(0, 0, 50, 100), box(50, 0, 100, 100)],
        index=gpd.pd.Index(["w", "e"], name="geoid"),
        crs=CRS,
    )
    bldgs = gpd.GeoDataFrame(
        {"height": [3.0, 27.0, 9.0]},
        geometry=[box(0, 0, 10, 10), box(80, 0, 90, 10), box(40, 50, 60, 60)],
        crs=CRS,
    )
    # T1/T2 split at x=45, so C's western half is itself split between targets
    targets = gpd.GeoDataFrame(
        geometry=[box(0, 0, 45, 100), box(45, 0, 100, 100)],
        index=["T1", "T2"],
        crs=CRS,
    )
    return tracts, bldgs, targets


def test_stack_footprints_one_row_per_floor():
    gdf = gpd.GeoDataFrame(
        {"height": [3.0, 27.0, np.nan, 1.0, 7.0]},
        geometry=[box(i, 0, i + 1, 1) for i in range(5)],
        index=list("abcde"),
        crs=CRS,
    )
    floors = stack_footprints(gdf, "height", 3)
    # 1, 9, 1 (missing height), 1 (below one floor), 2
    assert floors.index.value_counts().loc[list("abcde")].tolist() == [1, 9, 1, 1, 2]
    assert floors.loc["b", "floor"].tolist() == list(range(9))
    assert floors.is_valid.all()
    assert floors.area.sum() == pytest.approx(14.0)


def test_stack_footprints_floor_height():
    gdf = gpd.GeoDataFrame({"height": [3.0]}, geometry=[box(0, 0, 1, 1)], crs=CRS)
    with pytest.raises(ValueError, match="floor_height"):
        stack_footprints(gdf, "height", 0)


@pytest.mark.parametrize("allocate_total", [True, False])
def test_interpolate_weights_by_floors(one_tract, allocate_total):
    tract, bldgs, targets = one_tract
    out = floor_area_interpolate(
        tract,
        targets,
        bldgs,
        height_col="height",
        extensive_variables=["pop"],
        allocate_total=allocate_total,
    )
    np.testing.assert_allclose(out["pop"].to_numpy(), [100.0, 900.0], rtol=1e-6)


def test_interpolate_without_height_weights_by_footprint(one_tract):
    tract, bldgs, targets = one_tract
    out = floor_area_interpolate(tract, targets, bldgs, extensive_variables=["pop"])
    np.testing.assert_allclose(out["pop"].to_numpy(), [500.0, 500.0], rtol=1e-6)


def test_interpolate_allocate_total(one_tract):
    tract, bldgs, targets = one_tract
    only_b = targets.loc[["B"]]
    kwargs = dict(height_col="height", extensive_variables=["pop"])
    kept = floor_area_interpolate(tract, only_b, bldgs, allocate_total=False, **kwargs)
    pushed = floor_area_interpolate(tract, only_b, bldgs, allocate_total=True, **kwargs)
    assert kept["pop"].iloc[0] == pytest.approx(900.0, rel=1e-6)
    assert pushed["pop"].iloc[0] == pytest.approx(1000.0, rel=1e-6)


def test_interpolate_straddling_building(two_tracts):
    tracts, bldgs, targets = two_tracts
    out = floor_area_interpolate(
        tracts,
        targets,
        bldgs,
        height_col="height",
        extensive_variables=["pop"],
        intensive_variables=["rate"],
    )
    # T1: 1000 * (100 + 150) / 400; T2: 1000 * 150 / 400 + 600
    np.testing.assert_allclose(out["pop"].to_numpy(), [625.0, 975.0], rtol=1e-6)
    # T2 floor-area-weighted mean: (150 * 10 + 1200 * 20) / 1350
    np.testing.assert_allclose(out["rate"].to_numpy(), [10.0, 25500 / 1350], rtol=1e-6)
    assert out["pop"].sum() == pytest.approx(tracts["pop"].sum(), rel=1e-6)


def test_interpolate_requires_variables(one_tract):
    tract, bldgs, targets = one_tract
    with pytest.raises(ValueError, match="at least one"):
        floor_area_interpolate(tract, targets, bldgs)


def test_interpolate_duplicate_index(one_tract):
    tract, bldgs, targets = one_tract
    doubled = gpd.pd.concat([tract, tract])
    with pytest.raises(ValueError, match="duplicates"):
        floor_area_interpolate(doubled, targets, bldgs, extensive_variables=["pop"])


def test_dots_weight_by_floors_even_floor_count():
    """Points split 1:2 between a 1-floor and a 2-floor building."""
    tract = gpd.GeoDataFrame(
        {"pop": [30000]}, geometry=[box(0, 0, 100, 100)], index=["t"], crs=CRS
    )
    one, two = box(0, 0, 10, 10), box(20, 0, 30, 10)
    bldgs = gpd.GeoDataFrame({"height": [3.0, 6.0]}, geometry=[one, two], crs=CRS)
    pts = floor_area_dot_density(tract, bldgs, ["pop"], height_col="height", rng=0)
    assert len(pts) == 30000
    assert pts.within(two).mean() == pytest.approx(2 / 3, abs=0.015)


def test_dots_without_height_weight_by_footprint(one_tract):
    tract, bldgs, _ = one_tract
    tract = tract.assign(pop=20000)
    pts = floor_area_dot_density(tract, bldgs, ["pop"], rng=0)
    assert pts.within(bldgs.geometry["B"]).mean() == pytest.approx(0.5, abs=0.015)


@pytest.mark.parametrize("scale", [1.0, 0.2])
def test_dots_exact_counts_per_tract(two_tracts, scale):
    """Each tract and category gets round(count * scale) points, even when a
    category is far smaller than the number of floors it is spread over."""
    tracts, bldgs, _ = two_tracts
    # many small 1-floor buildings, so each floor holds a fraction of a person
    small = [box(x, y, x + 2, y + 2) for x in range(2, 98, 6) for y in range(20, 98, 6)]
    bldgs = gpd.pd.concat(
        [
            bldgs,
            gpd.GeoDataFrame({"height": [3.0] * len(small)}, geometry=small, crs=CRS),
        ],
        ignore_index=True,
    )
    tracts = tracts.assign(pop=[1000, 601], small=[7, 3])
    pts = floor_area_dot_density(
        tracts, bldgs, ["pop", "small"], height_col="height", scale=scale, rng=0
    )
    counts = pts.groupby(["geoid", "category"], observed=True).size()
    for geoid in tracts.index:
        for col in ["pop", "small"]:
            assert counts[(geoid, col)] == round(tracts.loc[geoid, col] * scale)
    assert pts.within(bldgs.union_all()).all()


def test_dots_report_source_index(two_tracts):
    tracts, bldgs, _ = two_tracts
    pts = floor_area_dot_density(tracts, bldgs, ["pop"], height_col="height", rng=0)
    assert "geoid" in pts.columns
    assert set(pts["geoid"]) == {"w", "e"}


def test_dots_warn_for_tract_without_buildings(two_tracts):
    tracts, bldgs, _ = two_tracts
    empty = gpd.GeoDataFrame(
        {"pop": [50.0], "rate": [1.0]},
        geometry=[box(200, 0, 300, 100)],
        index=gpd.pd.Index(["x"], name="geoid"),
        crs=CRS,
    )
    tracts = gpd.pd.concat([tracts, empty])
    with pytest.warns(UserWarning, match="contain no buildings"):
        pts = floor_area_dot_density(tracts, bldgs, ["pop"], height_col="height")
    assert "x" not in set(pts["geoid"])


def test_dots_reproject_buildings(one_tract):
    tract, bldgs, _ = one_tract
    pts_same = floor_area_dot_density(tract, bldgs, ["pop"], height_col="height", rng=1)
    pts_reproj = floor_area_dot_density(
        tract, bldgs.to_crs(4326), ["pop"], height_col="height", rng=1
    )
    assert pts_reproj.crs == tract.crs
    assert len(pts_same) == len(pts_reproj) == 1000


def test_floor_rows_match_overlay(two_tracts):
    tracts, bldgs, _ = two_tracts
    # add a building on the tract boundary and an invalid (self-crossing) one
    bowtie = gpd.GeoSeries.from_wkt(["POLYGON ((20 20, 30 30, 30 20, 20 30, 20 20))"])
    extra = gpd.GeoDataFrame(
        {"height": [6.0, 6.0]},
        geometry=[box(45, 0, 50, 5), bowtie.iloc[0]],
        crs=CRS,
    )
    bldgs = gpd.pd.concat([bldgs, extra], ignore_index=True)
    src = gpd.GeoDataFrame(
        {_SOURCE: np.arange(len(tracts))}, geometry=tracts.geometry.values, crs=CRS
    )
    expected = gpd.overlay(src, bldgs[["height", "geometry"]], keep_geom_type=True)
    got = _floor_rows(tracts, bldgs)
    assert len(got) == len(expected)
    by_unit = got.area.groupby(got[_SOURCE]).sum()
    assert by_unit.to_numpy() == pytest.approx(
        expected.area.groupby(expected[_SOURCE]).sum().to_numpy()
    )


def test_stack_footprints_floors_col():
    gdf = gpd.GeoDataFrame(
        {
            "height": [30.0, 9.0, np.nan, np.nan, 9.0],
            "floors": [2.0, np.nan, 4.7, np.nan, 0.0],
        },
        geometry=[box(i, 0, i + 1, 1) for i in range(5)],
        index=list("abcde"),
        crs=CRS,
    )
    count = lambda f: f.index.value_counts().loc[list("abcde")].tolist()  # noqa: E731
    # floor counts only: missing counts as one, 4.7 rounds down, zero becomes one
    assert count(stack_footprints(gdf, floors_col="floors")) == [2, 1, 4, 1, 1]
    # both: the recorded count wins where present, height fills the gaps
    both = stack_footprints(gdf, "height", 3, floors_col="floors")
    assert count(both) == [2, 3, 4, 1, 1]


def test_stack_footprints_needs_a_column():
    gdf = gpd.GeoDataFrame({"height": [3.0]}, geometry=[box(0, 0, 1, 1)], crs=CRS)
    with pytest.raises(ValueError, match="height_col"):
        stack_footprints(gdf)


def test_floors_col_matches_equivalent_height(one_tract):
    tract, bldgs, targets = one_tract
    by_height = floor_area_interpolate(
        tract, targets, bldgs, height_col="height", extensive_variables=["pop"]
    )
    by_floors = floor_area_interpolate(
        tract,
        targets,
        bldgs.assign(floors=bldgs["height"] / 3),
        floors_col="floors",
        extensive_variables=["pop"],
    )
    assert by_floors["pop"].tolist() == pytest.approx(by_height["pop"].tolist())
    assert by_floors["pop"].tolist() == pytest.approx([100, 900])


def test_dot_density_floors_col(one_tract):
    tract, bldgs, _ = one_tract
    pts = floor_area_dot_density(
        tract, bldgs.assign(floors=[1, 2]), ["pop"], floors_col="floors", rng=0
    )
    in_b = pts.geometry.x > 50
    assert len(pts) == 1000
    assert in_b.mean() == pytest.approx(2 / 3, abs=0.05)


def test_stack_footprints_default_floors():
    gdf = gpd.GeoDataFrame(
        {"height": [9.0, np.nan, np.nan], "floors": [np.nan, 4.0, np.nan]},
        geometry=[box(i, 0, i + 1, 1) for i in range(3)],
        index=list("abc"),
        crs=CRS,
    )
    stacked = stack_footprints(gdf, "height", 3, floors_col="floors", default_floors=2)
    # only c has neither a height nor a floor count
    assert stacked.index.value_counts().loc[list("abc")].tolist() == [3, 4, 2]
    with pytest.raises(ValueError, match="default_floors"):
        stack_footprints(gdf, "height", default_floors=0)


def test_default_floors_reweights_missing(one_tract):
    tract, bldgs, targets = one_tract
    # B's height is unknown: one floor by default gives 1:1, two floors gives 1:2
    bldgs = bldgs.assign(height=[3.0, np.nan])
    kw = dict(height_col="height", extensive_variables=["pop"])
    est = floor_area_interpolate(tract, targets, bldgs, **kw)
    assert est["pop"].tolist() == pytest.approx([500, 500])
    est = floor_area_interpolate(tract, targets, bldgs, default_floors=2, **kw)
    assert est["pop"].tolist() == pytest.approx([1000 / 3, 2000 / 3])
