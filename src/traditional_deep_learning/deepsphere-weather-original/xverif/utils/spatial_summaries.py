#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:46:04 2023.

@author: ghiggi
"""
import numpy as np

##----------------------------------------------------------------------------.
## Weighting for equiangular
# weights_lat = np.cos(np.deg2rad(lat))
# weights_lat /= weights_lat.sum()
# error * weights_lat


def global_summary(ds, area_coords="area"):
    """Compute global statistics weighted by grid cell area."""
    # Check area_coords
    area_weights = ds[area_coords] / ds[area_coords].values.sum()
    sample_dims = list(area_weights.dims)
    ds_weighted = ds.weighted(area_weights)
    return ds_weighted.mean(sample_dims)


def latitudinal_summary(ds, lat_dim="lat", lon_dim="lon", lat_res=5):
    """Compute latitudinal (bin) statistics, averaging over longitude."""
    # Check lat_dim and lon_dim
    # Check lat_res < 90
    # TODO: lon between -180 and 180 , lat between -90 and 90
    sample_dims = list(ds[lon_dim].dims)
    bins = np.arange(-90, 90 + lat_res, step=lat_res)
    labels = bins[:-1] + lat_res / 2
    return ds.groupby_bins(lat_dim, bins, labels=labels).mean(sample_dims)


def longitudinal_summary(ds, lat_dim="lat", lon_dim="lon", lon_res=5):
    """Compute longitudinal (bin) statistics, averaging over latitude."""
    # Check lat_dim and lon_dim
    # Check lon_res < 180
    # TODO: lon between -180 and 180 , lat between -90 and 90
    sample_dims = list(ds[lon_dim].dims)
    bins = np.arange(-180, 180 + lon_res, step=lon_res)
    labels = bins[:-1] + lon_res / 2
    return ds.groupby_bins(lon_dim, bins, labels=labels).mean(sample_dims)
