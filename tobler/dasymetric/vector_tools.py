"""Dasymetric interpolation using building footprints and heights.

Each building footprint is repeated once per floor, so a source unit's
population is spread over its residential *floor area* rather than its land
area. The floors are ordinary, valid polygons (perfectly overlapping copies of
one footprint, one row per floor), so the existing area-weighted machinery in
tobler handles them without modification.
"""

from warnings import warn

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pandas.api.types import is_list_like

from ..area_weighted import area_interpolate
from ..area_weighted.area_interpolate import (
    _area_tables_binning,
    _area_tables_binning_parallel,
)
from ..util import dot_density
from ..util.util import _inf_check, _nan_check

__all__ = ["stack_footprints", "floor_area_interpolate", "floor_area_dot_density"]

_SOURCE = "__source_position"


def _count_floors(
    gdf, height_col=None, floors_col=None, floor_height=3.0, default_floors=1
):
    """Number of floors in each footprint, at least one.

    A recorded floor count is used where present, and ``height // floor_height``
    otherwise; a footprint with neither gets ``default_floors``.
    """
    if height_col is None and floors_col is None:
        raise ValueError("pass `height_col`, `floors_col`, or both")
    if default_floors < 1:
        raise ValueError("`default_floors` must be at least one")
    n_floors = np.full(len(gdf), np.nan)
    if floors_col is not None:
        n_floors = np.floor(gdf[floors_col].to_numpy(dtype=float, na_value=np.nan))
    if height_col is not None:
        if floor_height <= 0:
            raise ValueError("`floor_height` must be greater than zero")
        heights = gdf[height_col].to_numpy(dtype=float, na_value=np.nan)
        n_floors = np.where(np.isnan(n_floors), heights // floor_height, n_floors)
    n_floors = np.where(np.isnan(n_floors), default_floors, n_floors)
    return np.maximum(1, n_floors).astype(int)


def stack_footprints(
    gdf, height_col=None, floor_height=3.0, floors_col=None, default_floors=1
):
    """Repeat each footprint once per floor.

    Counts the floors of each footprint, from a column of floor counts, a
    column of heights, or both, then copies each row that many times. The
    copies are identical, valid polygons, one row per floor, so the total area
    of the result is the total floor area of the input.

    Parameters
    ----------
    gdf : GeoDataFrame
        polygon footprints with a column holding the height or number of floors
        of each footprint.
    height_col : str, optional
        name of the column holding height (or any column describing the assumed
        number of stacked units in each footprint, in the units of
        ``floor_height``). Each footprint gets ``max(1, height // floor_height)``
        floors.
    floor_height : float
        floor-to-floor height assumed for every footprint, in the units of
        ``height_col``. Default is 3.0 since many datasets give height in meters.
    floors_col : str, optional
        name of the column holding the number of floors in each footprint,
        rounded down to a whole number. If both columns are given, a
        footprint's floor count is used where present and its height otherwise.
    default_floors : int
        number of floors given to a footprint with neither a height nor a floor
        count. Default is 1. Footprints with fewer than one floor count as one.

    Returns
    -------
    GeoDataFrame
        one row per floor, with the columns and (repeated) index of ``gdf`` and
        a ``floor`` column numbering each footprint's floors from 0.
    """
    n_floors = _count_floors(gdf, height_col, floors_col, floor_height, default_floors)
    floors = gdf.iloc[np.repeat(np.arange(len(gdf)), n_floors)].copy()
    first_row = np.repeat(np.cumsum(n_floors) - n_floors, n_floors)
    floors["floor"] = np.arange(len(floors)) - first_row
    return floors


def _floor_rows(
    source_df,
    buildings,
    height_col=None,
    floors_col=None,
    floor_height=3.0,
    default_floors=1,
):
    """Clip buildings to the source units, then repeat each piece once per floor.

    Clipping before stacking means no floor straddles two source units, so each
    floor row belongs to exactly one source unit (recorded positionally in the
    ``_SOURCE`` column).

    The result is ``gpd.overlay(source_df, buildings, how="intersection")``.
    Most buildings lie inside a single source unit, where the intersection is
    the building itself, so only the buildings that meet more than one unit (or
    are invalid) go through ``overlay``.
    """
    if source_df.crs is None or buildings.crs is None:
        raise ValueError("`source_df` and `buildings` must both have a CRS")
    if buildings.crs != source_df.crs:
        buildings = buildings.to_crs(source_df.crs)
    cols = [c for c in dict.fromkeys((height_col, floors_col)) if c is not None]
    bldgs = gpd.GeoDataFrame(
        buildings[cols].reset_index(drop=True),
        geometry=buildings.geometry.values,
        crs=source_df.crs,
    )
    src = gpd.GeoDataFrame(
        {_SOURCE: np.arange(len(source_df))},
        geometry=source_df.geometry.values,
        crs=source_df.crs,
    )

    # a building that meets one source unit and lies inside it is its own piece
    _, meets = bldgs.sindex.query(src.geometry, predicate="intersects")
    n_units = np.bincount(meets, minlength=len(bldgs))
    unit, inside = bldgs.sindex.query(src.geometry, predicate="contains")
    owner = np.full(len(bldgs), -1)
    owner[inside] = unit
    valid_unit = shapely.is_valid(np.asarray(src.geometry.values))
    whole = (n_units == 1) & (owner >= 0)
    whole[whole] = valid_unit[owner[whole]]
    whole[whole] = shapely.is_valid(np.asarray(bldgs.geometry.values)[whole])

    columns = [_SOURCE, *cols, bldgs.geometry.name]
    pieces = bldgs[whole].assign(**{_SOURCE: owner[whole]})[columns]
    if not whole.all():
        clipped = gpd.overlay(
            src, bldgs[~whole], how="intersection", keep_geom_type=True
        )
        pieces = pd.concat([pieces, clipped[columns]], ignore_index=True)
    if height_col is None and floors_col is None:
        floors = pieces.assign(floor=0)
    else:
        floors = stack_footprints(
            pieces, height_col, floor_height, floors_col, default_floors
        )
    return floors.reset_index(drop=True)


def _allocate_to_floors(source_df, floors, extensive=(), intensive=(), fill_nan=0.0):
    """Attach source values to floor rows.

    Extensive values are split by each floor's share of its source unit's total
    floor area; intensive values are copied to every floor. This is the result
    of ``area_interpolate(source_df, floors)``, computed directly because every
    floor lies inside exactly one source unit.
    """
    floors = floors.copy()
    pos = floors[_SOURCE].to_numpy()
    area = floors.area.to_numpy()
    total = np.bincount(pos, weights=area, minlength=len(source_df))
    share = area / total[pos]
    for variable in extensive:
        vals = _nan_check(source_df, variable, fill_value=fill_nan)
        vals = _inf_check(vals, variable, fill_value=fill_nan)
        floors[variable] = vals[pos] * share
    for variable in intensive:
        vals = _nan_check(source_df, variable, fill_value=fill_nan)
        vals = _inf_check(vals, variable, fill_value=fill_nan)
        floors[variable] = vals[pos]

    if extensive:
        empty = total == 0
        lost = source_df.loc[empty, list(extensive)].fillna(0).abs().sum(axis=1) > 0
        if lost.any():
            warn(
                f"{lost.sum()} source unit(s) have nonzero values in "
                f"{list(extensive)} but contain no buildings; their values are "
                "not allocated.",
                UserWarning,
                stacklevel=3,
            )
    return floors


def _check_source(source_df):
    if source_df.index.duplicated().any():
        raise ValueError("The index of the source_df cannot contain duplicates.")


def floor_area_interpolate(
    source_df,
    target_df,
    buildings,
    height_col=None,
    floors_col=None,
    floor_height=3.0,
    default_floors=1,
    extensive_variables=None,
    intensive_variables=None,
    allocate_total=True,
    n_jobs=-1,
    fill_nan=0.0,
):
    """Interpolate between polygon datasets, weighting by building floor area.

    Each source unit's values are spread over the floor area of the buildings
    inside it (footprint area times number of floors), then transferred to the
    target geometries with standard area-weighted interpolation.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        source data to be converted to another geometric representation.
    target_df : geopandas.GeoDataFrame
        target geometries that will form the new representation of the input data.
    buildings : geopandas.GeoDataFrame
        building footprints used as the dasymetric layer, already filtered to the
        buildings that should receive population (e.g. residential or unknown).
        Reprojected to the CRS of ``source_df`` if necessary.
    height_col : str, optional
        column of ``buildings`` holding building height. Each building gets
        ``max(1, height // floor_height)`` floors.
    floors_col : str, optional
        column of ``buildings`` holding the number of floors in each building,
        rounded down to a whole number. If both ``height_col`` and ``floors_col``
        are given, a building's floor count is used where present and its height
        otherwise. If both are None (the default), every building is a single
        floor, which weights by footprint area alone.
    floor_height : float
        floor-to-floor height in the units of ``height_col``. Default is 3.0.
    default_floors : int
        number of floors given to a building with neither a height nor a floor
        count, when ``height_col`` or ``floors_col`` is given. Default is 1.
    extensive_variables : list
        columns of ``source_df`` holding extensive variables (e.g. counts).
    intensive_variables : list
        columns of ``source_df`` holding intensive variables (e.g. rates). The
        estimate for each target is the floor-area-weighted mean of the source
        values.
    allocate_total : bool
        whether to allocate each source unit's full value to the targets even
        when they cover only part of its floor area, as in
        :func:`area_interpolate`. If False, the share of a unit's floor area
        outside every target is not allocated. Default is True.
    n_jobs : int
        number of processes used to build the area table. Default is -1 for all cores.
    fill_nan : numeric, str, or None
        value replacing NaN in the source variables; see :func:`area_interpolate`.
        Default is 0.0.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with the geometries of ``target_df`` and the interpolated
        variables as columns.

    Notes
    -----
    Source units that hold nonzero extensive values but contain no buildings
    raise a warning, and their values are not allocated.
    """
    extensive = list(extensive_variables or [])
    intensive = list(intensive_variables or [])
    if not (extensive or intensive):
        raise ValueError("pass at least one extensive or intensive variable")
    _check_source(source_df)

    floors = _floor_rows(
        source_df, buildings, height_col, floors_col, floor_height, default_floors
    )
    floors = _allocate_to_floors(source_df, floors, extensive, intensive, fill_nan)
    if target_df.crs != floors.crs:
        floors = floors.to_crs(target_df.crs)

    if n_jobs == 1:
        table = _area_tables_binning(floors, target_df, "auto")
    else:
        table = _area_tables_binning_parallel(floors, target_df, n_jobs=n_jobs)

    if allocate_total and extensive:
        # allocate each *source unit's* total to the floor area the targets cover,
        # as area_interpolate would for the source unit itself
        pos = floors[_SOURCE].to_numpy()
        n = len(source_df)
        covered = np.bincount(
            pos, weights=np.asarray(table.sum(axis=1)).ravel(), minlength=n
        )
        total = np.bincount(pos, weights=floors.area.to_numpy(), minlength=n)
        factor = np.divide(total, covered, out=np.zeros(n), where=covered > 0)
        floors[extensive] = floors[extensive].mul(factor[pos], axis=0)

    return area_interpolate(
        floors,
        target_df,
        extensive_variables=extensive or None,
        intensive_variables=intensive or None,
        table=table,
        allocate_total=False,
        fill_nan=fill_nan,
    )


def floor_area_dot_density(
    source_df,
    buildings,
    columns,
    height_col=None,
    floors_col=None,
    floor_height=3.0,
    default_floors=1,
    scale=1.0,
    rng=None,
):
    """Simulate points inside buildings, weighting by building floor area.

    Each source unit's counts are placed in the buildings inside it, with each
    floor of each building receiving points in proportion to its share of the
    unit's total floor area. Every source unit receives exactly
    ``round(count * scale)`` points per column.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        polygons holding the counts to simulate.
    buildings : geopandas.GeoDataFrame
        building footprints used as the dasymetric layer, already filtered to the
        buildings that should receive population (e.g. residential or unknown).
        Reprojected to the CRS of ``source_df`` if necessary.
    columns : list-like
        columns of ``source_df`` holding the number of points to simulate in each
        category, e.g. counts of mutually exclusive racial groups.
    height_col : str, optional
        column of ``buildings`` holding building height. Each building gets
        ``max(1, height // floor_height)`` floors.
    floors_col : str, optional
        column of ``buildings`` holding the number of floors in each building,
        rounded down to a whole number. If both ``height_col`` and ``floors_col``
        are given, a building's floor count is used where present and its height
        otherwise. If both are None (the default), every building is a single
        floor.
    floor_height : float
        floor-to-floor height in the units of ``height_col``. Default is 3.0.
    default_floors : int
        number of floors given to a building with neither a height nor a floor
        count, when ``height_col`` or ``floors_col`` is given. Default is 1.
    scale : float, optional
        coefficient applied to each source unit's counts, e.g. 0.2 to draw one
        point per five people. Default is 1.0.
    rng : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        random generator or seed.

    Returns
    -------
    geopandas.GeoDataFrame
        one row per simulated point, holding the index of the source unit that
        contains it and the column (category) it was drawn for.

    Notes
    -----
    Source units that hold nonzero counts but contain no buildings raise a
    warning and receive no points.
    """
    if not is_list_like(columns):
        raise ValueError("`columns` should be a list of columns")
    columns = list(columns)
    _check_source(source_df)

    floors = _floor_rows(
        source_df, buildings, height_col, floors_col, floor_height, default_floors
    )
    floors = _allocate_to_floors(source_df, floors, extensive=columns)
    floors.index = source_df.index[floors[_SOURCE].to_numpy()]
    floors.index.name = source_df.index.name

    return dot_density(floors, columns=columns, scale=scale, rng=rng, by=_SOURCE)
