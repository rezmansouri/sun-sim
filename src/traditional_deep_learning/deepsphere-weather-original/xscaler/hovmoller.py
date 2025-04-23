#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:33:55 2022

@author: ghiggi
"""
import xarray as xr
import numpy as np
from .checks import (
    check_variable_dim,
    check_spatial_dim,
    check_bin_width,
    check_bin_edges,
    check_time_dim,
    check_time_groups,
    get_time_groupby_idx,
)


def HovmollerDiagram(
    data,
    spatial_dim,
    time_dim,
    bin_edges=None,
    bin_width=None,
    time_groups=None,
    time_average_before_binning=True,
    variable_dim=None,
):
    """
    Compute an Hovmoller diagram.

    Parameters
    ----------
    data : xr.Data.Array or xr.Data.Array
        Either xr.Data.Array or xr.Dataset.
    spatial_dim : str
        The name of the spatial dimension over which to average values.
    time_dim : str
        The name of the time dimension.
    bin_edges : (list, np.ndarray), optional
        The bin edges over which to aggregate values across the spatial dimension.
        If not specified, bin_width must be specified.
    bin_width : (int, float), optional
        This argument is required if 'bin_edges' is not specified.
        Bins with 'bin_width' are automatically defined based 'spatial_dim' data range.
    time_groups : TYPE, optional
        DESCRIPTION. The default is None.
    time_average_before_binning : bool, optional
        If 'time_groups' is provided, wheter to average data over time groups before
        or after computation of the Hovmoller diagram.
        The default is True.
    variable_dim : str, optional
        If data is a DataArray, 'variable_dim' is used to reshape the tensor to
        an xr.Dataset with as variables the values of 'variable_dim'
        This allows to compute the statistic for each 'variable_dim' value.

    Returns
    -------
    xr.Data.Array or xr.Data.Array
        An Hovmoller diagram.

    """
    ##----------------------------------------------------------------.
    # Check data is an xarray Dataset or DataArray
    if not (isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray)):
        raise TypeError("'data' must be an xarray Dataset or xarray DataArray.")

    # - Checks for Dataset
    if isinstance(data, xr.Dataset):
        # Check variable_dim is not specified !
        if variable_dim is not None:
            raise ValueError(
                "'variable_dim' must not be specified for Dataset objects. Use groupby_dims instead."
            )

    # - Checks for DataArray (and convert to Dataset)
    flag_DataArray = False
    if isinstance(data, xr.DataArray):
        flag_DataArray = True
        da_name = data.name
        # Check variable_dim
        if variable_dim is None:
            # If not specified, data name will become the dataset variable name
            data = data.to_dataset()
        else:
            variable_dim = check_variable_dim(variable_dim=variable_dim, data=data)
            data = data.to_dataset(dim=variable_dim)
    ##----------------------------------------------------------------.
    # - Check for spatial_dim
    spatial_dim = check_spatial_dim(spatial_dim, data)

    # - If spatial_dim is not a dimension-coordinate, swap dimensions
    dims = np.array(list(data.dims))
    if not np.all(np.isin(spatial_dim, dims)):
        dim_tuple = data[spatial_dim].dims
        if len(dim_tuple) != 1:
            raise ValueError(
                "{} 'spatial_dim' coordinate must be 1-dimensional.".format(spatial_dim)
            )
        data = data.swap_dims({dim_tuple[0]: spatial_dim})
    ##----------------------------------------------------------------.
    # - Check for bin_width and bin_edges
    if bin_edges is None and bin_width is None:
        raise ValueError(
            "If 'bin_edges' are not specified, specify the desired 'bin_width'."
        )
    bin_width = check_bin_width(bin_width)
    # - Define bin_edges if not provided
    min_val = data[spatial_dim].min().values
    max_val = data[spatial_dim].max().values
    tol = 1.0e-8
    if bin_edges is None:
        bin_edges = np.arange(min_val, max_val + tol, bin_width)
    # - Define bin midpoints
    midpoints = bin_edges[:-1] + np.ediff1d(bin_edges) * 0.5
    # - Extend outermost edges to ensure min and max values to be included
    bin_edges[0] -= tol
    bin_edges[-1] += tol
    # - Check bin_edges validity (at least 2 bins)
    bin_edges = check_bin_edges(bin_edges, lb=min_val, ub=max_val)
    ##----------------------------------------------------------------.
    # Check time_dim
    time_dim = check_time_dim(time_dim=time_dim, data=data)
    if time_dim == spatial_dim:
        raise ValueError("'spatial_dim' can not be equal to 'time_dim'.")
    ##----------------------------------------------------------------.
    # Check time_groups
    time_groups = check_time_groups(time_groups=time_groups)

    ##----------------------------------------------------------------.
    # Retrieve indexing for temporal groupby
    time_groupby_info = get_time_groupby_idx(
        data=data, time_dim=time_dim, time_groups=time_groups
    )
    time_groupby_idx = time_groupby_info["time_groupby_idx"]
    time_groupby_dims = time_groupby_info["time_groupby_dims"]

    ##-----------------------------------------------------------------.
    # Optional aggregation over time before binning by spatial_dim
    if time_average_before_binning and time_groups is not None:
        data = data.groupby(time_groupby_idx).mean(time_dim)

    ##-----------------------------------------------------------------.
    # Compute average across spatial dimension bins
    hovmoller = (
        data.groupby_bins(spatial_dim, bin_edges, right=True)
        .mean(spatial_dim)
        .compute()
    )
    hovmoller[spatial_dim + "_bins"] = midpoints

    ##-----------------------------------------------------------------.
    # Optional aggregation over time after binning
    if not time_average_before_binning and time_groups is not None:
        hovmoller = hovmoller.groupby(time_groupby_idx).mean(time_dim)

    ##----------------------------------------------------------------.
    ## Remove non-dimension (Time_GroupBy) coordinate
    if time_groups is not None:
        time_groups_vars = list(time_groups.keys())
        for time_group in time_groups_vars:
            hovmoller[time_group] = time_groupby_dims[time_group]
            hovmoller = hovmoller.set_coords(time_group)
        if len(time_groups_vars) == 1:
            hovmoller = hovmoller.swap_dims(
                {time_groupby_idx.name: time_groups_vars[0]}
            )
        hovmoller = hovmoller.drop(time_groupby_idx.name)

    ##--------------------------------------------------------------------.
    # Reshape to DataArray if new_data was a DataArray
    if flag_DataArray:
        if variable_dim is None:
            return (
                hovmoller.to_array(dim="variable", name=da_name)
                .squeeze()
                .drop("variable")
            )
        else:
            return hovmoller.to_array(dim=variable_dim, name=da_name)
    else:
        return hovmoller

    ##----------------------------------------------------------------------------.
