#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:59:00 2022

@author: ghiggi
"""
import numpy as np
import xarray as xr

# TODO: lower case since a function
# Or convert to class !!!!


def OneHotEnconding(data, n_categories=None):
    """
    Perform OneHotEnconding of a categorical xarray DataArray.

    Parameters
    ----------
    data : xr.DataArray
        xarray DataArray to OneHotEncode.
    n_categories : int, optional
        Specify the number of categories. The default is None.

    Returns
    -------
    Returns an xarray Dataset with OneHotEncoding variables.

    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("'data' must be a xarray DataArray.")
    if not isinstance(n_categories, (int, type(None))):
        raise TypeError("'n_categories' must be an integer.")
    ##------------------------------------------------------------------------.
    # Convert data as integers
    x = data.values.astype(int)
    # Compute n_categories
    if n_categories is None:
        n_categories = np.max(x) + 1
    else:
        min_n_categories = np.max(x) + 1
        if n_categories < min_n_categories:
            raise ValueError(
                "'n_categories' must be equal or larger than {}.".format(
                    min_n_categories
                )
            )
    ##------------------------------------------------------------------------.
    # Compute OHE tensor
    OHE = np.eye(n_categories)[x]
    ##------------------------------------------------------------------------.
    # Create Dataset
    da_name = data.name
    list_da = []
    for cat in range(n_categories):
        tmp_da = data.copy()
        tmp_da.values = OHE[..., cat]
        tmp_da.name = da_name + " (OHE Class " + str(cat) + ")"
        list_da.append(tmp_da)
    ds = xr.merge(list_da)
    ##------------------------------------------------------------------------.
    return ds


def InvertOneHotEnconding(data, name=None):
    """
    Invert OneHotEnconded variables of an xarray Dataset.

    Parameters
    ----------
    data : xr.Dataset
        xarray Dataset with OneHotEncoded variables
    name: str
        Name of the output xarray DataArray
    Returns
    -------
    Returns an xarray DataArray with categorical labels.

    """
    if not isinstance(data, xr.Dataset):
        raise TypeError("'data' must be a xarray DataArray.")
    if not isinstance(name, (str, type(None))):
        raise TypeError("'name' must be a string (or None).")
    ##------------------------------------------------------------------------.
    # Convert Dataset to numpy tensor
    OHE = data.to_array("OHE").transpose(..., "OHE").values

    (OHE > 1).any() or (OHE < 0).any()
    # Check all values are between 0 or 1 (so that works also for probs)
    if (OHE > 1).any() or (OHE < 0).any():
        raise ValueError("Expects all values to be between 0 and 1")
    ##-----------------------------------------------------------------------.
    # Inverse
    x = np.argmax(OHE, axis=len(OHE.shape) - 1)
    ##------------------------------------------------------------------------.
    # Create DataArray
    da = data[list(data.data_vars.keys())[0]]
    da.values = x
    da.name = name
    ##------------------------------------------------------------------------.
    return da
