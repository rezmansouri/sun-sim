#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:39:34 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr

#### TODO: move in utils.forecast.py


def time_at_inittime_leadtime(
    obj: xr.DataArray | xr.Dataset,
    init_time_name: str = "forecast_reference_time",
    lead_time_name: str = "lead_time",
):
    """Compute a 2d array of the actual time given the initialization and lead times."""
    reftime = obj[init_time_name].values
    leadtime = obj[lead_time_name].values

    if (dtype := leadtime.dtype) != np.dtype("timedelta64[ns]"):
        raise ValueError(
            f"{lead_time_name} coordinate must be of dtype 'np.timedelta' but is {dtype}"
        )

    return xr.DataArray(
        reftime[:, None] + leadtime[None],
        coords={init_time_name: reftime, lead_time_name: leadtime},
        dims=[init_time_name, lead_time_name],
    )


def temporal_broadcast_observations(
    obs: xr.DataArray | xr.Dataset,
    fcts: xr.DataArray | xr.Dataset,
    obs_time_name: str = "time",
    fcts_init_time_name: str = "forecast_reference_time",
    fcts_lead_time_name: str = "lead_time",
):
    """Align temporal coordinates of observations to those of forecasts.

    Conform an observation object that has a single temporal dimension,
    representing the actual time of the observation, to a forecast object
    that has two temporal dimensions representing the forecast initialization
    time and the lead time (i.e. the temporal hozizon of the forecast).

    Parameters
    ----------
    obs: xr.DataArray or xr.Dataset
        The observation object, where the temporal dimension must have a coordinate
        array of dtype `np.datetime64[ns]`.
    fcts: xr.DataArray or xr.Dataset
        The forecast object, where the forecast initialization and lead time dimensions must have
        coordinate arrays of dtype `np.datetime64[ns]` and `np.timedelta64[ns]` respectively.
    obs_time_tame: str
        The name of the temporal dimension on the observation object.
    fcts_init_time_name: str
        The name of the initialization time dimension on the forecast object.
    fcts_lead_time_name: str
        The name of the lead time dimension on the forecast object.

    Returns
    -------
    obs: xr.DataArray or xr.Dataset
    """
    indexer = time_at_inittime_leadtime(fcts, fcts_init_time_name, fcts_lead_time_name)
    try:
        obs = obs.sel({obs_time_name: indexer})
    except KeyError:
        obs = obs.reindex(time=np.unique(indexer.values.reshape(-1)))
        obs = obs.sel(time=indexer)
    fcts = fcts.assign_coords({obs_time_name: indexer})
    return obs, fcts
