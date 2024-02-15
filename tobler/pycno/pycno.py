"""Pycnophylactic Interpolation (contributed by @danlewis85)."""
# https://github.com/danlewis85/pycno/

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from numpy import (
    absolute,
    apply_along_axis,
    asarray,
    convolve,
    copy,
    nan,
    nanmax,
    nanmean,
    nansum,
    pad,
    power,
    round,
    unique,
)
from numpy.ma import masked_invalid, masked_where
from pandas import DataFrame
from rasterio.features import rasterize


def pycno(
    gdf, value_field, cellsize, r=0.2, handle_null=True, converge=3, verbose=True
):
    """Returns a smooth pycnophylactic interpolation raster for a given geodataframe

    Args:
    gdf (geopandas.geodataframe.GeoDataFrame): Input GeoDataFrame.
    value_field (str): Field name of values to be used to produce pycnophylactic surface
    cellsize (int): Pixel size of raster in planar units (i.e. metres, feet)
    r (float, optional): Relaxation parameter, default of 0.2 is generally fine.
    handle_null (boolean, optional): Changes how nodata values are smoothed. Default True.
    converge (int, optional): Index for stopping value, default 3 is generally fine.
    verbose (boolean, optional): Print out progress at each iteration.

    Returns:
    Numpy Array: Smooth pycnophylactic interpolation.
    Rasterio geotransform
    GeoPandas crs
    """
    # set nodata value
    nodata = -9999

    # work out raster rows and columns based on gdf extent and cellsize
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xres = int((xmax - xmin) / cellsize)
    yres = int((ymax - ymin) / cellsize)

    # Work out transform so that we rasterize the area where the data are!
    trans = rasterio.Affine.from_gdal(xmin, cellsize, 0, ymax, 0, -cellsize)

    # First make a zone array
    # NB using index values as ids can often be too large/alphanumeric. Limit is int32 polygon features.
    # create a generator of geom, index pairs to use in rasterizing
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.index))
    # burn the features into a raster array
    feature_array = rasterize(
        shapes=shapes, fill=nodata, out_shape=(yres, xres), transform=trans
    )

    # Get cell counts per index value (feature)
    unique, count = np.unique(feature_array, return_counts=True)
    cellcounts = asarray((unique, count)).T
    # Lose the nodata counts
    cellcounts = cellcounts[cellcounts[:, 0] != nodata, :]
    # Adjust value totals by cells
    # Make cell counts dataframe
    celldf = DataFrame(cellcounts[:, 1], index=cellcounts[:, 0], columns=["cellcount"])
    # Merge cell counts
    gdf = gdf.merge(celldf, how="left", left_index=True, right_index=True)
    # Calculate cell values
    gdf["cellvalues"] = gdf[value_field] / gdf["cellcount"]

    # create a generator of geom, cellvalue pairs to use in rasterizing
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.cellvalues))
    # Now burn the initial value raster
    value_array = rasterize(
        shapes=shapes, fill=nodata, out_shape=(yres, xres), transform=trans
    )

    # Set nodata in value array to np.nan
    value_array[value_array == -9999] = nan

    # Set stopper value based on converge parameter
    stopper = nanmax(value_array) * power(10.0, -converge)

    # The basic numpy convolve function doesn't handle nulls.
    def smooth2D(data):
        # Create function that calls a 1 dimensionsal smoother.
        s1d = lambda s: convolve(s, [0.5, 0.0, 0.5], mode="same")
        # pad the data array with the mean value
        padarray = pad(data, 1, "constant", constant_values=nanmean(data))
        # make nodata mask
        mask = masked_invalid(padarray).mask
        # set nodata as zero to avoid eroding the raster
        padarray[mask] = 0.0
        # Apply the convolution along each axis of the data and average
        padarray = (
            apply_along_axis(s1d, 1, padarray) + apply_along_axis(s1d, 0, padarray)
        ) / 2
        # Reinstate nodata
        padarray[mask] = nan
        return padarray[1:-1, 1:-1]

    # The convolution function from astropy handles nulls.
    def astroSmooth2d(data):
        try:
            from astropy.convolution import convolve as astro_convolve
        except (ImportError, ModuleNotFoundError) as err:
            raise ImportError(
                "Pycnophylactic interpolation with handle_null=True "
                "requires the astropy package"
            ) from err
        s1d = lambda s: astro_convolve(s, [0.5, 0, 0.5])
        # pad the data array with the mean value
        padarray = pad(data, 1, "constant", constant_values=nanmean(data))
        # Apply the convolution along each axis of the data and average
        padarray = (
            apply_along_axis(s1d, 1, padarray) + apply_along_axis(s1d, 0, padarray)
        ) / 2
        return padarray[1:-1, 1:-1]

    def correct2Da(data):
        for idx, val in gdf[value_field].items():
            # Create zone mask from feature_array
            mask = masked_where(feature_array == idx, feature_array).mask
            # Work out the correction factor
            correct = (val - nansum(data[mask])) / mask.sum()
            # Apply correction
            data[mask] += correct

        return data

    def correct2Dm(data):
        for idx, val in gdf[value_field].items():
            # Create zone mask from feature_array
            mask = masked_where(feature_array == idx, feature_array).mask
            # Work out the correction factor
            correct = val / nansum(data[mask])
            if correct != 0.0:
                # Apply correction
                data[mask] *= correct

        return data

    while True:
        # Store the current iteration
        old = copy(value_array)

        # Smooth the value_array
        if handle_null:
            sm = astroSmooth2d(value_array)
        else:
            sm = smooth2D(value_array)

        # Relaxation to prevent overcompensation in the smoothing step
        value_array = value_array * r + (1.0 - r) * sm

        # Perform correction
        value_array = correct2Da(value_array)

        # Reset any negative values to zero.
        value_array[value_array < 0] = 0.0

        # Perform correction
        value_array = correct2Dm(value_array)

        if verbose:
            print(
                "Maximum Change: "
                + str(round(nanmax(absolute(old - value_array)), 4))
                + " - will stop at "
                + str(round(stopper, 4))
            )

        if nanmax(absolute(old - value_array)) < stopper:
            break

    return (value_array, trans, gdf.crs)


def save_pycno(pycno_array, transform, crs, filestring, driver="GTiff"):
    """Saves a numpy array as a raster, largely a helper function for pycno
    Args:
        pycno_array (numpy array): 2D numpy array of pycnophylactic surface
        transform (rasterio geotransform): Relevant transform from pycno()
        crs (GeoPandas crs): Coordinate reference system of GeoDataFrame used in pycno()
        filestring (str): File path to save raster
        driver (str, optional): Format for output raster, default: geoTiff.
    Returns:
        None
    """
    import rasterio

    # Save raster
    new_dataset = rasterio.open(
        filestring,
        "w",
        driver=driver,
        height=pycno_array.shape[0],
        width=pycno_array.shape[1],
        count=1,
        dtype="float64",
        crs=crs,
        transform=transform,
    )
    new_dataset.write(pycno_array.astype("float64"), 1)
    new_dataset.close()

    return None


def extract_values(pycno_array, gdf, transform, fieldname="Estimate"):
    """Extract raster value sums according to a provided polygon geodataframe
    Args:
        pycno_array (numpy array): 2D numpy array of pycnophylactic surface.
        gdf (geopandas.geodataframe.GeoDataFrame): Target GeoDataFrame.
        transform (rasterio geotransform): Relevant transform from pycno()
        fieldname (str, optional): New gdf field to save estimates in. Default name: 'Estimate'.
    Returns:
        geopandas.geodataframe.GeoDataFrame: Target GeoDataFrame with appended estimates.
    """
    from numpy import nansum
    from rasterio.features import geometry_mask

    estimates = []
    # Iterate through geodataframe and extract values
    for idx, geom in gdf["geometry"].items():
        mask = geometry_mask(
            [geom], pycno_array.shape, transform=transform, invert=True
        )
        estimates.append(nansum(pycno_array[mask]))
    out = pd.Series(estimates, index=gdf.index)
    return out


def pycno_interpolate(
    source_df,
    target_df,
    variables,
    cellsize,
    r=0.2,
    handle_null=True,
    converge=3,
    verbose=False,
):
    """Pycnophylactic Inerpolation.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame (required)
        geodataframe with polygon geometries and data to transfer
    target_df : geopandas.GeoDataFrame (required)
        geodataframe with polygon geometries to receive new data
    variables : list
        columns on the source_df containing data to transfer
    cellsize : int
        Pixel size of intermediate raster in planar units (i.e. metres, feet)
    r : float, optional
        Relaxation parameter, default of 0.2 is generally fine
    handle_null : bool, optional
        Changes how nodata values are smoothed. Default True.
    converge : int, optional
        Index for stopping value, default 3 is generally fine.
    verbose : bool, optional
        Print out progress at each iteration.

    Returns
    -------
    geopandas.GeoDataFrame
        new geodataframe with interpolated variables as columns and target_df geometry
        as output geometry

    Notes
    -----
    The formula is based on Tobler, W. R. (1979). Smooth pycnophylactic interpolation for geographical regions. Journal of the American Statistical Association, 74(367), 519–529. https://doi.org/10.1080/01621459.1979.10481647

    Original implementation written by @danlewis85 at <https://github.com/danlewis85/pycno/>
    and based in part on the R pycno package by Chris Brusndon (<https://cran.r-project.org/web/packages/pycno/index.html>)

    References: :cite:`tobler_smooth_1979`
    """
    assert source_df.crs.equals(
        target_df.crs
    ), "source_df CRS and target_df CRS are not the same. Reproject into consistent systems before proceeding"
    output_vars = target_df.copy()[[target_df.geometry.name]]
    for variable in variables:
        pyc, trans, _ = pycno(
            source_df,
            variable,
            cellsize=cellsize,
            r=r,
            handle_null=handle_null,
            converge=converge,
            verbose=verbose,
        )
        vals = extract_values(pyc, target_df, transform=trans)
        output_vars[variable] = vals

    return output_vars
