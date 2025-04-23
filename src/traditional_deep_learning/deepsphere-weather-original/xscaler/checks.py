#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:34:46 2022

@author: ghiggi
"""
import xarray as xr
import numpy as np
import os

#### Utils ####


def get_valid_scaler_class():
    """Return list of implemented xscaler objects."""
    scaler_classes = [
        "GlobalStandardScaler",
        "TemporalStandardScaler",
        "GlobalMinMaxScaler",
        "TemporalMinMaxScaler",
        "SequentialScaler",
    ]
    return scaler_classes


def check_valid_scaler(scaler):
    """Check is a valid scaler."""
    # TODO : Check type/class instead looking for attribute...
    if scaler.scaler_class not in get_valid_scaler_class():
        print(scaler.scaler_class)
        raise ValueError("Not a valid scaler")


def check_variable_dim(variable_dim, data):
    """Check that the correct variable dimension (for DataArray) is specified."""
    # Check type
    if variable_dim is None:
        return None
    if not isinstance(variable_dim, str):
        raise TypeError("Provide 'variable_dim' as a string")
    # Check validity
    dims = list(data.dims)
    if variable_dim not in dims:
        raise ValueError(
            "'variable_dim' must be a dimension coordinate of the xarray object"
        )
    # Return variable_dim as a string
    return variable_dim


def check_groupby_dims(groupby_dims, data):
    """Check that valid groupby dimensions are specified."""
    # Check type
    if isinstance(groupby_dims, str):
        groupby_dims = [groupby_dims]
    if not (isinstance(groupby_dims, list) or isinstance(groupby_dims, tuple)):
        raise TypeError("Provide 'groupby_dims' as a string, list or tuple")
    # Check validity
    dims = np.array(list(data.dims))
    if not np.all(np.isin(groupby_dims, dims)):
        raise ValueError(
            "'groupby_dims' must be dimension coordinates of the xarray object"
        )
    # Return grouby_dims as a list of strings
    return groupby_dims


def check_rename_dict(data, rename_dict):
    """Check rename_dict validity."""
    if not isinstance(rename_dict, dict):
        raise ValueError("'rename_dict' must be a dictionary.")
    data_dims = list(data.dims)
    keys = list(rename_dict.keys())
    vals = list(rename_dict.values())
    keys_all = np.all(np.isin(keys, data_dims))
    vals_all = np.all(np.isin(vals, data_dims))
    if keys_all:
        new_dict = rename_dict
    elif vals_all:
        new_dict = {v: k for k, v in rename_dict.items()}
    else:
        raise ValueError(
            "The specified dimensions in 'rename_dict' are not dimensions of the supplied data."
        )
    return new_dict


def get_xarray_variables(data, variable_dim=None):
    """Return the variables of an xarray Dataset or DataArray."""
    if isinstance(data, xr.Dataset):
        return list(data.data_vars.keys())
    elif isinstance(data, xr.DataArray):
        if variable_dim is None:
            return data.name
        else:
            variable_dim = check_variable_dim(variable_dim=variable_dim, data=data)
            return data[variable_dim].values.tolist()
    else:
        raise TypeError("Provide an xarray Dataset or DataArray")


def _check_is_fitted(self):
    # TODO: this could be assigned to a superclass of scalers
    if not self.fitted:
        raise ValueError("Please fit() the scaler before saving it!")


def _check_save_fpath(fpath, force):
    # Check basepath exists
    if not os.path.exists(os.path.dirname(fpath)):
        # If not exist, create directory
        os.makedirs(os.path.dirname(fpath))
        print(
            "The directory {} did not exist and has been created !".format(
                os.path.dirname(fpath)
            )
        )
    # Check end with .nc
    if fpath[-3:] != ".nc":
        fpath = fpath + ".nc"
        print("Added .nc extension to the provided fpath.")
    # If the netCDF file already exists, remove if force=True
    if os.path.exists(fpath):
        if force:
            os.remove(fpath)
        else:
            raise ValueError(
                "{} already exists on disk. Set force=True to overwrite.".format(fpath)
            )
    return fpath


# -----------------------------------------------------------------------------.
# ################################
#### Utils for TemporalScalers ###
# ################################
def check_time_dim(time_dim, data):
    """Check that the correct time dimension is specified."""
    # Check type
    if not isinstance(time_dim, str):
        raise TypeError("Specify 'time_dim' as a string.")
    # Check validity
    dims = list(data.dims)
    if time_dim not in dims:
        raise ValueError(
            "'time_dim' must specify the time dimension coordinate of the xarray object."
        )
    if not isinstance(data[time_dim].values[0], np.datetime64):
        raise ValueError(
            "'time_dim' must indicate a time dimension coordinate with np.datetime64 dtype."
        )
    # Return time_dim as a string
    return time_dim


def get_valid_time_groups():
    """Return valid time groups."""
    time_groups = [
        "year",
        "season",
        "quarter",
        "month",
        "day",
        "weekofyear",
        "dayofweek",
        "dayofyear",
        "hour",
        "minute",
        "second",
    ]
    return time_groups


def get_dict_season():
    """Return dictionary for conversion (to integers) of season strings."""
    dict_season = {
        "DJF": 1,
        "MAM": 2,
        "JJA": 3,
        "SON": 4,
    }
    return dict_season


def get_time_group_max(time_group):
    """Return dictionary with max values for each time group."""
    dict_time_max = {
        "year": 5000,  # dummy large value for year since unbounded ...
        "season": 4,
        "quarter": 4,
        "month": 12,
        "weekofyear": 53,
        "dayofweek": 7,  # init 0, end 6
        "dayofyear": 366,
        "day": 31,
        "hour": 24,  # init 0, end 23
        "minute": 60,  # init 0, end 59
        "second": 60,  # init 0, end 59
    }
    return dict_time_max[time_group]


def get_time_group_min(time_group):
    """Return dictionary with min values for each time group."""
    dict_time_min = {
        "year": 0,
        "season": 1,
        "quarter": 1,
        "month": 1,
        "weekofyear": 1,
        "dayofweek": 0,
        "dayofyear": 1,
        "day": 1,
        "hour": 0,
        "minute": 0,
        "second": 0,
    }
    return dict_time_min[time_group]


def get_time_groupby_name(time_groups):
    """Return a name reflecting the temporal groupby operation."""
    # Define time groupby name
    time_groups_list = []
    for k, v in time_groups.items():
        if v == 1:
            time_groups_list.append(k)
        else:
            time_groups_list.append(str(v) + k)
    time_groupby_name = "Time_GroupBy " + "-".join(time_groups_list)
    return time_groupby_name


def check_time_groups(time_groups):
    """Check validity of time_groups."""
    if time_groups is None:
        return None
    # Check type
    if isinstance(time_groups, str):
        time_groups = [time_groups]
    if isinstance(time_groups, list):
        time_groups = {k: 1 for k in time_groups}
    if not isinstance(time_groups, dict):
        raise TypeError("Provide time_groups as string, list or dictionary.")
    ##------------------------------------------------------------------------.
    # Check time_groups name validity
    time_groups_name = np.array(list(time_groups.keys()))
    unvalid_time_groups_name = time_groups_name[
        np.isin(time_groups_name, get_valid_time_groups(), invert=True)
    ]
    if len(unvalid_time_groups_name) > 0:
        raise ValueError(
            "{} are not valid 'time_groups' keys".format(unvalid_time_groups_name)
        )
    ##------------------------------------------------------------------------.
    # Check time_groups time aggregation validity
    for k, v in time_groups.items():
        # Check min value
        if v < 1:
            raise ValueError(
                "The aggregation period of '{}' must be at least 1".format(k)
            )
        # Check max value
        max_val = get_time_group_max(time_group=k)
        if v > get_time_group_max(time_group=k):
            raise ValueError(
                "The maximum aggregation period of '{}' is {}".format(k, max_val)
            )
        # Check max value is divisible by specified time aggregation
        if (max_val % v) != 0:
            print(
                "Attention, the specified aggregation period ({}) does not allow uniform subdivision of '{}'".format(
                    v, k
                )
            )
    ##------------------------------------------------------------------------.
    return time_groups


def check_time_groupby_factors(time_groupby_factors, time_groups):
    """Check validity of time_groupby_factors."""
    if time_groupby_factors is None:
        return {}
    if time_groups is not None:
        if not np.all(np.isin(time_groups.keys(), time_groupby_factors.keys())):
            raise ValueError(
                "All time groups must be included in time_groupby_factors."
            )
        return time_groupby_factors
    else:
        return {}


def check_new_time_groupby_idx(time_groupby_idx, scaler_stat):
    """Check that the fitted scaler contains all time_groupby_idx of new_data."""
    time_groupby_idx_orig = np.unique(scaler_stat[time_groupby_idx.name].values)
    time_groupby_idx_new = np.unique(time_groupby_idx.values)
    if not np.all(np.isin(time_groupby_idx_new, time_groupby_idx_orig)):
        raise ValueError(
            "The TemporalScaler does not contain representative statistics for all time_groups indices of 'new_data'."
        )


##----------------------------------------------------------------------------.
def get_time_groupby_idx(data, time_dim, time_groups, time_groupby_factors=None):
    """Return a 1D array with unique index for temporal groupby operation."""
    # Check time groups
    time_groups_dict = check_time_groups(time_groups=time_groups)
    # Check time_groupby_factors
    time_groupby_factors = check_time_groupby_factors(
        time_groupby_factors, time_groups=time_groups_dict
    )
    no_time_groupby_factors = len(time_groupby_factors) == 0
    ##------------------------------------------------------------------------.
    # Retrieve groupby indices
    if time_groups is not None:
        tmp_min_interval = 0
        l_time_groups_dims = []
        for i, (time_group, time_agg) in enumerate(time_groups_dict.items()):
            # Retrieve max time aggregation
            time_agg_max = get_time_group_max(time_group=time_group)
            # Retrieve time index (for specific time group)
            # -  dt.week, dt.weekofyear has been deprecated in Pandas ...
            if time_group == "weekofyear":
                idx = data[time_dim].dt.isocalendar().week
            else:
                idx = data[time_dim].dt.__getattribute__(time_group)
            l_time_groups_dims.append(idx)
            # Preprocessing if 'season' (string to integer)
            if time_group == "season":
                dict_season = get_dict_season()
                idx_values = [dict_season[s] for s in idx.values]
                idx.values = idx_values
            ##----------------------------------------------------------------.
            # Define (numeric) indexing for groupby
            idx_agg = np.floor(
                idx / time_agg
            )  # set equal indices within time_agg period
            idx_norm = idx_agg / (time_agg_max / time_agg)  # value between 0 and 1
            ##----------------------------------------------------------------.
            if no_time_groupby_factors:
                # get_numeric_combo_factor()
                if tmp_min_interval == 0:
                    idx_scaled = idx_norm  # *10⁰
                    tmp_min_interval = np.max(np.unique(idx_scaled))
                    time_groupby_factors[time_group] = 0  # 10^0 = 1
                    time_groupby_idx = idx_scaled
                else:
                    factor = 0
                    while True:
                        idx_scaled = idx_norm * (10**factor)
                        unique_idx = np.unique(idx_scaled)
                        if np.min(np.diff(unique_idx)) > tmp_min_interval:
                            break
                        else:
                            factor = factor + 1
                    tmp_min_interval = tmp_min_interval + np.max(unique_idx)
                    time_groupby_idx = time_groupby_idx + idx_scaled
                    time_groupby_factors[time_group] = factor
            else:
                idx_scaled = idx_norm * (10 ** time_groupby_factors[time_group])
                if i == 0:
                    time_groupby_idx = idx_scaled
                else:
                    time_groupby_idx = time_groupby_idx + idx_scaled
        ##--------------------------------------------------------------------.
        # Add name to time groupby indices
        time_groupby_idx_name = get_time_groupby_name(time_groups_dict)
        time_groupby_idx.name = time_groupby_idx_name
        # Retrieve time_groups_dims coords
        time_groups_dims = xr.merge(l_time_groups_dims)
        # Retrieve unique extended time_groupby_dims coords
        time_groupby_dims = xr.merge([time_groupby_idx, time_groups_dims])
        _, index = np.unique(
            time_groupby_dims[time_groupby_idx_name], return_index=True
        )
        time_groupby_dims = time_groupby_dims.isel({time_dim: index})
        time_groupby_dims = time_groupby_dims.swap_dims(
            {time_dim: time_groupby_idx_name}
        ).drop(time_dim)
    # If no time_groups are specified --> Long-term mean
    else:
        # Set all indices to 0 (unique group over time --> long-term mean)
        time_groupby_idx = data.time.dt.month
        time_groupby_idx[:] = 0
        time_groupby_idx.name = "Long-term mean"
        time_groupby_dims = None
    ##------------------------------------------------------------------------.
    # Create time_groupby info dictionary
    time_groupby_info = {}
    time_groupby_info["time_groupby_idx"] = time_groupby_idx
    time_groupby_info["time_groupby_idx_name"] = time_groupby_idx.name
    time_groupby_info["time_groupby_factors"] = time_groupby_factors
    time_groupby_info["time_groupby_dims"] = time_groupby_dims
    return time_groupby_info


##----------------------------------------------------------------------------.
def check_reference_period(reference_period):
    """Check reference_period validity."""
    # Check type
    if reference_period is None:
        return None
    if not isinstance(reference_period, (list, tuple, np.ndarray)):
        raise TypeError(
            "'reference period' must be either a list, tuple or numpy array with start and end time period."
        )
    if len(reference_period) != 2:
        raise ValueError(
            "'reference period' require 2 elements: start time and end time."
        )
    ##------------------------------------------------------------------------.
    # If np.array with np.datetime64
    if isinstance(reference_period, np.ndarray):
        if not np.issubdtype(reference_period.dtype, np.datetime64):
            raise ValueError("If a numpy array, must have np.datetime64 dtype.")
        else:
            return reference_period
    ##------------------------------------------------------------------------.
    if isinstance(reference_period, (list, tuple)):
        try:
            reference_period = np.array(reference_period, dtype="M8")
        except ValueError:
            raise ValueError(
                "The values of reference_period can not be converted to datetime64."
            )
    return reference_period


##----------------------------------------------------------------------------.
#### Utils for Hovmoller
def check_spatial_dim(spatial_dim, data):
    """Check that a valid spatial dimension is specified."""
    # Check type
    if not isinstance(spatial_dim, str):
        raise TypeError("Specify 'spatial_dim' as a string.")
    # Check validity
    coords = list(data.coords.keys())
    if spatial_dim not in coords:
        raise ValueError("'spatial_dim' must be a coordinate of the xarray object.")
    # Return spatial_dim as a list of strings
    return spatial_dim


##----------------------------------------------------------------------------.
def check_bin_width(bin_width):
    if not isinstance(bin_width, (int, float)):
        raise TypeError("'bin_width' must be an integer or float number.")
    if bin_width <= 0:
        raise ValueError("'bin_width' must be a positive number larger than 0.")
    return bin_width


##----------------------------------------------------------------------------.
def check_bin_edges(bin_edges, lb, ub):
    if not isinstance(bin_edges, (list, np.ndarray)):
        raise TypeError("'bin_edges' must be a list or numpy.ndarray.")
    if isinstance(bin_edges, list):
        bin_edges = np.array(bin_edges)
    # Select and sort only unique values
    bin_edges = np.sort(np.unique(bin_edges))
    # Check that at least 2 bins can be defined
    if len(bin_edges) < 3:
        raise ValueError("'bin_edges' must have minimum 3 unique values.")
    # Ensure that some data falls within the bins
    if bin_edges[0] >= ub:
        raise ValueError("The left edge exceed the max value.")
    if bin_edges[-1] <= lb:
        raise ValueError("The right edge exceed the min value.")
    n_bins_within_data_range = sum(np.logical_and(bin_edges > lb, bin_edges < ub))
    if n_bins_within_data_range < 2:
        raise ValueError(
            "Too much values in 'bin_edges' are outside data range to create at least 1 bin."
        )
    return bin_edges
