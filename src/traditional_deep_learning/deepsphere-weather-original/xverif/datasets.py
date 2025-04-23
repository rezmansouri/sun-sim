#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:14:51 2023.

@author: ghiggi
"""
import datetime

import numpy as np
import xarray as xr


def _get_random_data(shape, data_type, n_categories=2):
    """Generate random array based on data_type."""
    if data_type in ["continuous"]:
        data = np.random.rand(*shape)  # within [0, 1]
    elif data_type in ["probability"]:
        raise NotImplementedError("Need to adapt code for 2D array")
    elif data_type in ["multiclass", "binary"]:
        class_values = np.arange(0, n_categories)
        data = np.random.choice(class_values, size=shape)
    else:
        valid_data_type = [
            "continuous",
            "multiclass",
            "binary",
            "probability",
        ]  # TODO: get_valid_data_type()
        raise ValueError(f"Valid data_type are {valid_data_type}")
    return data


def _create_datarray(
    obs_number,
    obs_dim,
    data_type,
    aux_shape,
    aux_dims,
    n_categories,
    name,
):
    """Generate a DataArray."""
    # Check inputs
    if isinstance(aux_shape, int):
        aux_shape = [aux_shape]
    if isinstance(aux_dims, str):
        aux_dims = [aux_dims]

    # TODO: check positive shape value and integer
    if len(aux_dims) != len(aux_shape):
        raise ValueError("'aux_dims' and 'aux_shape' size must coincide.")
    if not isinstance(obs_number, int):
        raise ValueError("'obs_number' must be an integer'")

    # -------------------------------------------------------------------------.
    # Define array shape
    shape = [obs_number] + list(aux_shape)

    # Generate data
    data = _get_random_data(shape=shape, data_type=data_type, n_categories=n_categories)

    # Generate time coordinate
    start_date = datetime.datetime(1992, 7, 22)
    end_date = start_date + datetime.timedelta(days=obs_number)
    time_coord = np.arange(start_date, end_date, datetime.timedelta(days=1))

    # Generate other coordinates
    aux_coords = [np.arange(n) for n in aux_shape]

    # Define coordinates
    coords = [time_coord] + aux_coords

    # Define dimensions
    dims = [obs_dim] + list(aux_dims)

    # Generate DataArray
    da = xr.DataArray(data, coords=coords, dims=dims, name=name)
    return da


def _create_dataset(
    obs_number,
    obs_dim,
    data_type,
    aux_shape,
    aux_dims,
    n_categories,
    n_vars,
):
    """Generate a Dataset."""
    list_da = [
        _create_datarray(
            obs_number=obs_number,
            obs_dim=obs_dim,
            data_type=data_type,
            aux_shape=aux_shape,
            aux_dims=aux_dims,
            n_categories=n_categories,
            name="var" + str(i),
        )
        for i in range(n_vars)
    ]
    ds = xr.merge(list_da)
    return ds


def create_spatial2d_dataset(
    obs_number=1000,
    obs_dim="time",
    data_type="continuous",
    aux_shape=(10, 10),
    aux_dims=("x", "y"),
    n_categories=2,
    n_vars=3,
):
    """Create spatial 2D Dataset."""
    ds = _create_dataset(
        obs_number=obs_number,
        obs_dim=obs_dim,
        data_type=data_type,
        aux_shape=aux_shape,
        aux_dims=aux_dims,
        n_categories=n_categories,
        n_vars=n_vars,
    )
    return ds


def create_spatial3d_dataset(
    obs_number=1000,
    obs_dim="time",
    data_type="continuous",
    aux_shape=(10, 10, 10),
    aux_dims=("x", "y", "z"),
    n_categories=2,
    n_vars=3,
):
    """Create spatial 3D Dataset."""
    ds = _create_dataset(
        obs_number=obs_number,
        obs_dim=obs_dim,
        data_type=data_type,
        aux_shape=aux_shape,
        aux_dims=aux_dims,
        n_categories=n_categories,
        n_vars=n_vars,
    )
    return ds


def create_timeseries_dataset(
    obs_number=1000,
    obs_dim="time",
    data_type="continuous",
    aux_shape=(100),
    aux_dims=("stations"),
    n_categories=2,
    n_vars=3,
):
    """Create time series Dataset."""
    ds = _create_dataset(
        obs_number=obs_number,
        obs_dim=obs_dim,
        data_type=data_type,
        aux_shape=aux_shape,
        aux_dims=aux_dims,
        n_categories=n_categories,
        n_vars=n_vars,
    )
    return ds


def create_timeseries_forecast_dataset(
    obs_number=1000,
    obs_dim="time",
    data_type="continuous",
    aux_shape=(10, 48),
    aux_dims=("stations", "forecast_lead_time"),
    n_categories=2,
    n_vars=3,
):
    """Create time series forecast Dataset."""
    ds = _create_dataset(
        obs_number=obs_number,
        obs_dim=obs_dim,
        data_type=data_type,
        aux_shape=aux_shape,
        aux_dims=aux_dims,
        n_categories=n_categories,
        n_vars=n_vars,
    )
    return ds


def create_multimodel_dataset(
    obs_number=1000,
    obs_dim="time",
    data_type="continuous",
    aux_shape=(10, 10, 5),
    aux_dims=("x", "y", "model"),
    n_categories=2,
    n_vars=3,
):
    """Create multimodel Dataset."""
    ds = _create_dataset(
        obs_number=obs_number,
        obs_dim=obs_dim,
        data_type=data_type,
        aux_shape=aux_shape,
        aux_dims=aux_dims,
        n_categories=n_categories,
        n_vars=n_vars,
    )
    return ds


def create_ensemble_dataset(
    obs_number=1000,
    obs_dim="time",
    data_type="continuous",
    aux_shape=(10, 10, 24),
    aux_dims=("x", "y", "realization"),
    n_categories=2,
    n_vars=3,
):
    """Create ensemble Dataset."""
    ds = _create_dataset(
        obs_number=obs_number,
        obs_dim=obs_dim,
        data_type=data_type,
        aux_shape=aux_shape,
        aux_dims=aux_dims,
        n_categories=n_categories,
        n_vars=n_vars,
    )
    return ds


def create_ensemble_forecast_dataset(
    obs_number=1000,
    obs_dim="time",
    data_type="continuous",
    aux_shape=(10, 10, 24, 48),
    aux_dims=("x", "y", "realization", "leadtime"),
    n_categories=2,
    n_vars=3,
):
    """Create ensemble forecast Dataset."""
    ds = _create_dataset(
        obs_number=obs_number,
        obs_dim=obs_dim,
        data_type=data_type,
        aux_shape=aux_shape,
        aux_dims=aux_dims,
        n_categories=n_categories,
        n_vars=n_vars,
    )
    return ds


def create_multimodel_ensemble_forecast_dataset(
    obs_number=1000,
    obs_dim="time",
    data_type="continuous",
    aux_shape=(10, 10, 5, 24, 48),
    aux_dims=("x", "y", "model", "realization", "leadtime"),
    n_categories=2,
    n_vars=3,
):
    """Create multimodel ensemble forecast Dataset."""
    ds = _create_dataset(
        obs_number=obs_number,
        obs_dim=obs_dim,
        data_type=data_type,
        aux_shape=aux_shape,
        aux_dims=aux_dims,
        n_categories=n_categories,
        n_vars=n_vars,
    )
    return ds


#### For testing:
# obs_number=1000
# obs_dim="time"
# data_type="continuous"
# aux_shape=(10,10)
# aux_dims=["x", "y"]
# n_categories=3
# name="var"

# aux_shape=()
# aux_dims=()

# aux_shape=(1)
# aux_dims=('x')
