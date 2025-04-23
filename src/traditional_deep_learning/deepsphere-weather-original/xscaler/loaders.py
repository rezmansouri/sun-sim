#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:48:54 2022

@author: ghiggi
"""
import os
import xarray as xr
from .global_scalers import GlobalStandardScaler, GlobalMinMaxScaler
from .temporal_scalers import TemporalStandardScaler, TemporalMinMaxScaler
from .climatology import Climatology
from .anomaly import AnomalyScaler

# TODO: This should be a super class of the objects


def LoadScaler(fpath):
    """Load xarray scalers."""
    # Check .nc
    if fpath[-3:] != ".nc":
        fpath = fpath + ".nc"
    # Check exist
    if not os.path.exists(fpath):
        raise ValueError("{} does not exist on disk.".format(fpath))
    # Create scaler
    ds_scaler = xr.open_dataset(fpath)
    scaler_class = ds_scaler.attrs["scaler_class"]
    if scaler_class == "GlobalStandardScaler":
        return GlobalStandardScaler(data=None, ds_scaler=ds_scaler)
    if scaler_class == "GlobalMinMaxScaler":
        return GlobalMinMaxScaler(data=None, ds_scaler=ds_scaler)
    if scaler_class == "TemporalStandardScaler":
        return TemporalStandardScaler(data=None, time_dim=None, ds_scaler=ds_scaler)
    if scaler_class == "TemporalMinMaxScaler":
        return TemporalMinMaxScaler(data=None, time_dim=None, ds_scaler=ds_scaler)


def LoadClimatology(fpath):
    """Load Climatology object."""
    # Check .nc
    if fpath[-3:] != ".nc":
        fpath = fpath + ".nc"
    # Check exist
    if not os.path.exists(fpath):
        raise ValueError("{} does not exist on disk.".format(fpath))
    # Create scaler
    ds_clim = xr.open_dataset(fpath)
    return Climatology(data=None, time_dim=None, ds_climatology=ds_clim)


def LoadAnomaly(fpath):
    """Load xarray scalers.

    Useful because return an AnomalyScaler class... which allows to choose
    between anomalies and std anomalies.
    """
    # Check .nc
    if fpath[-3:] != ".nc":
        fpath = fpath + ".nc"
    # Check exist
    if not os.path.exists(fpath):
        raise ValueError("{} does not exist on disk.".format(fpath))
    # Create scaler
    ds_anomaly = xr.open_dataset(fpath)
    return AnomalyScaler(data=None, time_dim=None, ds_anomaly=ds_anomaly)
