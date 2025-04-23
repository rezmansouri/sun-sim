#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:48:21 2023.

@author: ghiggi
"""
import typing as tp
from importlib import import_module

import xarray as xr

from xverif.masking import mask_dataarrays, mask_datasets

ValidForecastType = tp.Literal[
    "continuous", "binary", "multiclass", "probability", "ordinal", "count"
]
ValidMetricType = tp.Literal["deterministic", "probabilistic", "spatial"]


def _check_args(pred, obs) -> None:
    if not (isinstance(pred, xr.DataArray) and isinstance(obs, xr.DataArray)):
        raise ValueError(
            "'pred' and 'obs' arguments must be xarray DataArray instances. Got "
            f"{type(pred)} and {type(obs)} types respectively."
        )
    if len(set(pred.dims) & set(obs.dims)) == 0:
        raise ValueError(
            "'pred' and 'obs' Datasets do not have any variable in common."
        )


def _check_shared_dims(dims, pred, obs):
    dims = set(dims)
    if dims < set(fdim := pred.dims) and dims < set(odim := obs.dims):
        pass
    else:
        raise ValueError(
            f"The aggregating dimensions {dims} must be present in both forecast "
            f"and observation objects, but they have {fdim} and {odim} respectively."
        )


def _check_data_type(data_type: str) -> None:
    if data_type not in (valid := tp.get_args(ValidForecastType)):
        raise ValueError(
            f"{data_type} is not a valid data type. Must be one of {valid}."
        )


def _check_sample_dims(sample_dims, obs):
    """Check sample_dims format."""
    if not isinstance(sample_dims, (type(None), str, list, tuple)):
        raise TypeError("'aggregatings_dims' must be a str, list, tuple or None.")
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    else:
        # Set defaults if None or empty list/tuple
        dims = list(obs.dims)
        if isinstance(sample_dims, type(None)):
            sample_dims = dims
        if len(sample_dims) == 0:
            sample_dims = dims
    return sample_dims


def _check_metric_type(metric_type):
    """Check metric_type validity."""
    if not isinstance(metric_type, str):
        raise TypeError("'metric_type' must be a string.")
    if metric_type not in tp.get_args(ValidMetricType):
        raise ValueError(f"Valid 'metric_type' are {tp.get_args(ValidMetricType)}")


def _get_xr_routine(metric_type, data_type, implementation="vectorized"):
    """Retrieve xarray routine to compute the metrics."""
    # Check inputs
    _check_data_type(data_type)
    _check_metric_type(metric_type)
    # Define module path
    # TODO: module_path = f"xverif.metrics.{metric_type}.{data_type}
    module_path = f"xverif.metrics.{metric_type}.{data_type}_{implementation}"
    # Import the module
    module = import_module(module_path)
    # Import the function
    function = module._xr_apply_routine
    return function


def ensure_common_xarray_format(pred, obs):
    """Ensure common format. Either xr.DataArray or xr.Dataset."""
    if not isinstance(pred, (xr.DataArray, xr.Dataset)):
        raise TypeError("'pred' is not an xarray object.")
    if not isinstance(obs, (xr.DataArray, xr.Dataset)):
        raise TypeError("'obs' is not an xarray object.")
    if isinstance(pred, xr.Dataset) and not isinstance(obs, xr.Dataset):
        raise TypeError("Both input datasets must be xr.Dataset or xr.DataArray.")
    if isinstance(pred, xr.DataArray) and not isinstance(obs, xr.DataArray):
        raise TypeError("Both input datasets must be xr.Dataset or xr.DataArray.")
    return pred, obs


def ensure_dataarray(xr_obj):
    """Convert xr.Dataset to DataArray.

    It expects that 'variable' is not a dimension of the xr.Dataset
    """
    if isinstance(xr_obj, xr.Dataset):
        if "variable" in list(xr_obj.dims):
            raise ValueError(
                "Please do not provide a xr.Dataset with a dimension named 'variable'."
            )
        xr_obj = xr_obj.to_array(dim="variable")
    return xr_obj


def align_xarray_objects(pred, obs):
    """Align xarray objects.

    - Ensure coordinate alignment for matching dimensions.
    - Ensure common subset of Dataset variables.
    """
    # Align xarray dimensions
    pred, obs = xr.align(pred, obs, join="inner")

    return pred, obs


def deterministic(
    pred,
    obs,
    data_type="continuous",
    sample_dims=None,
    implementation="vectorized",
    metrics=None,
    compute=True,
    skip_options=None,
    n_categories=None,
):
    """Compute deterministic metrics."""
    pred, obs = ensure_common_xarray_format(pred, obs)  # _check_args(pred, obs)

    sample_dims = _check_sample_dims(sample_dims, obs)
    _check_shared_dims(sample_dims, pred, obs)
    _check_data_type(data_type)

    # Define default skip_options options
    # TODO: - Set default based on data_type, ...
    if skip_options is None:
        skip_options = [{"nan": True}, {"inf": True}]

    # Check that obs dims is equal or subset of pred dims
    # TODO:

    # Check homogeneous datasets
    # TODO: and suggest function that return list of homogeneous datasets !

    # Align Datasets
    pred, obs = align_xarray_objects(pred, obs)

    # Apply masking for vectorized computations
    if implementation == "vectorized":
        # Broadcast obs to pred (for preprocessing)
        # - Creates a view, not a copy !
        obs = obs.broadcast_like(pred)
        # Mask Dataset / DataArrays
        if isinstance(pred, xr.Dataset):
            pred, obs = mask_datasets(pred, obs, masking_options=skip_options)
        else:
            pred, obs = mask_dataarrays(pred, obs, masking_options=skip_options)

    # Convert Dataset to DataArray
    # - Enable to vectorize also over variables if numpy
    pred = ensure_dataarray(pred)
    obs = ensure_dataarray(obs)

    # Define routine kwargs
    routine_kwargs = {}
    if implementation == "loop":
        routine_kwargs["drop_options"] = skip_options
    elif implementation == "vectorized":
        routine_kwargs["metrics"] = metrics
    if data_type == "multiclass":
        routine_kwargs["n_categories"] = n_categories

    # Retrieve xarray routine
    _xr_routine = _get_xr_routine(
        metric_type="deterministic",
        data_type=data_type,
        implementation=implementation,
    )

    # Compute skills
    ds_skill = _xr_routine(
        pred=pred,
        obs=obs,
        sample_dims=sample_dims,
        compute=compute,
        **routine_kwargs,
    )

    return ds_skill


def probabilistic():
    """Compute probabilistic skill metrics."""


def spatial():
    """Compute spatial skill metrics."""


# -----------------------------------------------------------------------------.
