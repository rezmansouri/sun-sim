#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:35:45 2022

@author: ghiggi
"""
import numpy as np
import xarray as xr


def check_node_dim(x, node_dim):
    """Check node dimension."""
    if not isinstance(node_dim, str):
        raise TypeError(
            "'node_dim' must be a string specifying the node dimension name."
        )
    if not isinstance(x, (xr.Dataset, xr.DataArray)):
        raise TypeError("You must provide an xarray Dataset or DataArray.")
    if isinstance(x, xr.Dataset):
        dims = list(x.dims.keys())
    if isinstance(x, xr.DataArray):
        dims = list(x.dims)
    if node_dim not in dims:
        raise ValueError("Specify the 'node_dim'.")
    return node_dim


def check_mesh(mesh):
    """Check the mesh format."""
    return mesh


def check_mesh_exist(x):
    """Check the mesh is available in the xarray object."""
    coords = list(x.coords.keys())
    if "mesh" not in coords:
        raise ValueError("No 'mesh' available in the xarray object.")


def check_mesh_area_exist(x, mesh_area_coord):
    """Check the area coordinate is available in the xarray object."""
    coords = list(x.coords.keys())
    if not isinstance(mesh_area_coord, str):
        raise TypeError("'area_coord' must be a string specifying the area coordinate.")
    if mesh_area_coord not in coords:
        raise ValueError(
            "No {} coordinate available in the xarray object.".format(mesh_area_coord)
        )


def check_valid_coords(x, coords):
    """Check coordinates validity."""
    if isinstance(coords, str):
        coords = [coords]
    if not isinstance(coords, list):
        raise TypeError("'coords' must be a string or a list of string.")
    valid_coords = list(x.coords.keys())
    not_valid = np.array(coords)[np.isin(coords, valid_coords, invert=True)]
    if len(not_valid) > 0:
        raise ValueError(
            "{} are not coordinates of the xarray object. Valid coordinates are {}.".format(
                not_valid, valid_coords
            )
        )


def check_xy(x_obj, x, y):
    """Check validty of x and y coordinates."""
    if not isinstance(x, str):
        raise TypeError("'x' must be a string indicating the longitude coordinate.")
    if not isinstance(y, str):
        raise TypeError("'x' must be a string indicating the latitude coordinate.")
    # Check x and y are coords of the xarray object
    check_valid_coords(x_obj, coords=[x, y])
