#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:03:14 2023.

@author: ghiggi
"""
import os
import shutil

import numpy as np
import xarray as xr

from xverif.utils.zarr import check_chunks, rechunk_Dataset


# ----------------------------------------------------------------------------.
def reshape_forecasts_for_verification(ds):
    """Process a Dataset with forecasts in the format required for verification."""
    l_reshaped_ds = []
    for i in range(len(ds["leadtime"])):
        tmp_ds = ds.isel(leadtime=i)
        tmp_ds["forecast_reference_time"] = (
            tmp_ds["forecast_reference_time"] + tmp_ds["leadtime"]
        )
        tmp_ds = tmp_ds.rename({"forecast_reference_time": "time"})
        l_reshaped_ds.append(tmp_ds)
    ds = xr.concat(l_reshaped_ds, dim="leadtime", join="outer")
    return ds


def rechunk_forecasts_for_verification(
    ds, target_store, chunks="auto", max_mem="1GB", force=False
):
    """
    Rechunk forecast Dataset in the format required for verification.

    Make data contiguous over the time dimension, and chunked over space.
    The forecasted time (referred as dimension 'time') is computed by
    summing the leadtime to the forecast_reference_time.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with dimensions 'forecast_reference_time' and 'leadtime'.
    target_store : TYPE
        Filepath of the zarr store where to save the new Dataset.
    chunks : str, optional
        Option for custom chunks of the new Dataset. The default is "auto".
        The default is chunked pixel-wise and per leadtime, contiguous over time.
    max_mem : str, optional
        The amount of memory (in bytes) that workers are allowed to use.
        The default is '1GB'.

    Returns
    -------
    ds_verification : xarray.Dataset
        Dataset for verification (with 'time' and 'leadtime' dimensions.

    """
    ##------------------------------------------------------------------------.
    # Check target_store do not exist already
    if os.path.exists(target_store):
        if force:
            shutil.rmtree(target_store)
        else:
            raise ValueError(
                "A zarr store already exists at {}. If you want to overwrite, specify force=True".format(
                    target_store
                )
            )
    ##------------------------------------------------------------------------.
    # Define temp store for rechunking
    temp_store = os.path.join(os.path.dirname(target_store), "tmp_store.zarr")
    # Define intermediate store for rechunked data
    intermediate_store = os.path.join(
        os.path.dirname(target_store), "rechunked_store.zarr"
    )

    ##------------------------------------------------------------------------.
    # Remove temp_store and intermediate_store is exists
    if os.path.exists(temp_store):
        shutil.rmtree(temp_store)
    if os.path.exists(intermediate_store):
        shutil.rmtree(intermediate_store)
    ##------------------------------------------------------------------------.
    # Default chunking
    # - Do not chunk along forecast_reference_time, chunk 1 to all other dimensions
    dims = list(ds.dims)
    dims_optional = np.array(dims)[
        np.isin(dims, ["time", "feature"], invert=True)
    ].tolist()
    default_chunks = {dim: 1 for dim in dims_optional}
    default_chunks["forecast_reference_time"] = -1
    default_chunks["leadtime"] = 1
    # Check chunking
    chunks = check_chunks(ds=ds, chunks=chunks, default_chunks=default_chunks)
    ##------------------------------------------------------------------------.
    # Rechunk Dataset (on disk)
    rechunk_Dataset(
        ds=ds,
        chunks=chunks,
        target_store=intermediate_store,
        temp_store=temp_store,
        max_mem=max_mem,
        force=force,
    )
    ##------------------------------------------------------------------------.
    # Load rechunked dataset (contiguous over forecast reference time, chunked over space)
    ds = xr.open_zarr(intermediate_store, chunks="auto")
    ##------------------------------------------------------------------------.
    # Reshape
    ds_verification = reshape_forecasts_for_verification(ds)
    ##------------------------------------------------------------------------.
    # Remove 'chunks' key in encoding (bug in xarray-dask-zarr)
    for var in list(ds_verification.data_vars.keys()):
        ds_verification[var].encoding.pop("chunks")

    ##------------------------------------------------------------------------.
    # Write to disk
    ds_verification.to_zarr(target_store)
    ##------------------------------------------------------------------------.
    # Remove rechunked store
    shutil.rmtree(intermediate_store)
    ##------------------------------------------------------------------------.
    # Load the Dataset for verification
    ds_verification = xr.open_zarr(target_store)
    ##------------------------------------------------------------------------.
    # Return the Dataset for verification
    return ds_verification


# ----------------------------------------------------------------------------.
