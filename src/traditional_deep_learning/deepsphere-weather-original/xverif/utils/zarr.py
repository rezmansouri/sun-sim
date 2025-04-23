#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:11:33 2021.

@author: ghiggi
"""
import os
import shutil

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from rechunker import rechunk

##########################
#### Compressor utils ####
##########################


def is_numcodecs(compressor):
    """Check is a numcodec compressor."""
    if type(compressor).__module__.find("numcodecs") == -1:
        return False
    return True


def _check_variables_names_type(variable_names):
    """Check variable_names type."""
    if not isinstance(variable_names, (list, str)):
        raise TypeError("'variable_names' must be a string or a list")
    if isinstance(variable_names, str):
        variable_names = [variable_names]
    if not all([isinstance(s, str) for s in variable_names]):
        raise TypeError(
            "Specify all variable names as string within the 'variable_names' list."
        )
    return variable_names


def _check_compressor_type(compressor, variable_names):
    """Check compressor type."""
    if not (
        isinstance(compressor, (str, dict, type(None))) or is_numcodecs(compressor)
    ):
        raise TypeError(
            "'compressor' must be a dictionary, numcodecs compressor, 'auto' string or None."
        )
    if isinstance(compressor, str):
        if not compressor == "auto":
            raise ValueError("If 'compressor' is specified as string, must be 'auto'.")
    if isinstance(compressor, dict):
        if not np.all(np.isin(list(compressor.keys()), variable_names)):
            raise ValueError(
                f"The 'compressor' dictionary must contain the keys {variable_names}"
            )
    return compressor


def _check_default_compressor_type(default_compressor, variable_names):
    """Check default_compressor type."""
    if not (
        isinstance(default_compressor, (dict, type(None)))
        or is_numcodecs(default_compressor)
    ):
        raise TypeError("'default_compressor' must be a numcodecs compressor or None.")
    if isinstance(default_compressor, dict):
        if not np.all(np.isin(list(default_compressor.keys()), variable_names)):
            raise ValueError(
                "The 'default_compressor' dictionary must contain the keys {variable_names}"
            )
    return default_compressor


def _check_compressor_dict(compressor):
    """Check compressor dictionary validity."""
    all_valid = [
        is_numcodecs(cmp) or isinstance(cmp, type(None)) for cmp in compressor.values()
    ]
    if not all(all_valid):
        raise ValueError(
            "All compressors specified in the 'compressor' dictionary must be numcodecs (or None)."
        )
    return compressor


def check_compressor(compressor, variable_names, default_compressor=None):
    """Check compressor validity for zarr writing.

    compressor = None --> No compression.
    compressor = "auto" --> Use default_compressor if specified. Otherwise will default to ds.to_zarr() default compressor.
    compressor = <numcodecs class> --> Specify the same compressor to all Dataset variables
    compressor = {..} --> A dictionary specifying a compressor for each Dataset variable.

    default_compressor: None or numcodecs compressor. None will default to ds.to_zarr() default compressor.
    variable_names: list of xarray Dataset variables
    """
    variable_names = _check_variables_names_type(variable_names)
    compressor = _check_compressor_type(compressor, variable_names)
    default_compressor = _check_compressor_type(default_compressor, variable_names)

    # If a string --> "Auto" --> Apply default_compressor (if specified)
    if isinstance(compressor, str):
        if compressor == "auto":
            compressor = default_compressor

    # If a unique compressor, create a dictionary with the same compressor for all variables
    elif is_numcodecs(compressor) or isinstance(compressor, type(None)):
        compressor = {var: compressor for var in variable_names}

    # If a dictionary, check keys validity and compressor validity
    else:  # isinstance(compressor, dict):
        compressor = _check_compressor_dict(compressor)

    return compressor


# -----------------------------------------------------------------------------.
######################
#### Chunks utils ####
######################


def _all_valid_chunks_values(values):
    """Checks chunks value validity."""
    bool_list = []
    for x in values:
        if isinstance(x, str):
            if x == "auto":
                bool_list.append(True)
            else:
                bool_list.append(False)
        elif isinstance(x, int):
            bool_list.append(True)
        elif isinstance(x, type(None)):
            bool_list.append(True)  # Require caution
        else:
            bool_list.append(False)
    return all(bool_list)


def _all_integer_chunks_values(chunks):
    """Check validity of chunks values.

    "auto", "None" and 0 are not allowed.
    """
    return all(isinstance(chunk, int) and chunk > 0 for chunk in chunks)


def get_dataset_chunks(ds):
    """Get dataset chunks dictionary."""
    variable_names = list(ds.data_vars.keys())
    chunks = {}
    for var in variable_names:
        if ds[var].chunks is not None:
            chunks[var] = {
                dim: v[0] for dim, v in zip(ds[var].dims, ds[var].chunks, strict=True)
            }
        else:
            chunks[var] = None
    return chunks


def _contains_all_keys(dictionary, keys):
    """Check a dictionary contains all the keys."""
    dict_keys = list(dictionary.keys())
    return np.all(np.isin(dict_keys, keys)) and np.all(np.isin(keys, dict_keys))


def _check_contains_all_keys(dictionary, keys, error_message):
    """Check a dictionary contains all the keys, else raise error."""
    if not _contains_all_keys(dictionary, keys):
        raise ValueError(error_message)


def _use_default_chunks(ds, default_chunks):
    """Define chunks if chunks='auto'.

    If default_chunks is a dict, set chunks=default_chunks
    If default_chunks = None, create chunks dictionary  with <dimension>:'auto'.
    """
    if isinstance(default_chunks, dict):
        chunks = default_chunks
    else:
        chunks = {dim: "auto" for dim in list(ds.dims)}
    return chunks


def _is_chunk_dict_per_variable(chunks_dict, ds):
    """Return True if chunks dictionary is defined per variable."""
    variable_names = list(ds.data_vars.keys())
    return _contains_all_keys(chunks_dict, variable_names)


def _is_chunk_dict_per_dims(chunks_dict, ds):
    """Return True if chunks dictionary is defined per dimension."""
    dim_names = list(ds.dims)
    return _contains_all_keys(chunks_dict, dim_names)


def _checks_chunks_type(chunks):
    """Check chunks input type."""
    # Check chunks
    if not isinstance(chunks, (str, dict, type(None))):
        raise TypeError("'chunks' must be a dictionary, 'auto' or None.")
    if isinstance(chunks, str):
        if not chunks == "auto":
            raise ValueError("If 'chunks' is specified as string, must be 'auto'.")
    return chunks


def _check_default_chunks(default_chunks, ds):
    """Check default chunks type and keys."""
    if not isinstance(default_chunks, (dict, type(None))):
        raise TypeError("'default_chunks' must be either a dictionary or None.")

    if not isinstance(default_chunks, type(None)):
        is_chunks_per_variable = _is_chunk_dict_per_variable(default_chunks, ds)
        is_chunks_per_dims = _is_chunk_dict_per_dims(default_chunks, ds)
        if not is_chunks_per_variable and not is_chunks_per_dims:
            # TODO: more info error on which dim or var is missing
            raise ValueError(
                "Invalid default_chunks. Must include all Dataset variables or dimensions."
            )
    return default_chunks


def _check_chunks_dict(chunks, ds):
    """Check chunks dictionaries validity.

    Return a 'per variable' chunks dictionary.
    """
    is_chunks_per_variable = _is_chunk_dict_per_variable(chunks, ds)
    is_chunks_per_dims = _is_chunk_dict_per_dims(chunks, ds)
    if not is_chunks_per_variable and not is_chunks_per_dims:
        # TODO: more info error on which dim or var is missing
        raise ValueError(
            "Invalid chunks. Must include all Dataset variables or dimensions."
        )

    if is_chunks_per_variable:
        chunks = _checks_variables_chunks_dict(ds, chunks)
    else:  #  is_chunks_per_dims:
        chunks = _convert_dims_chunks_dict(ds, chunks)  # to per variable
    return chunks


def _checks_variables_chunks_dict(ds, chunks):
    """Check validity of a chunk dictionary defined for each Dataset variable."""
    # - Check that for each Dataset variable, the chunks are specified for each dimension
    for var in list(ds.data_vars):
        da_dims = list(ds[var].dims)
        msg = f"The 'chunks' dictionary of {var} must contain the keys {da_dims}."
        _check_contains_all_keys(chunks[var], da_dims, msg)
    # - Check chunks value validity
    for var in list(ds.data_vars):
        if not _all_integer_chunks_values(list(chunks[var].values())):
            raise ValueError("Invalid 'chunks' values for {var}.")
    return chunks


def _convert_dims_chunks_dict(ds, chunks):
    """Convert 'per dimension' chunks dictionary to 'per variable' dictionary."""
    # - Checks chunks value validity
    if not _all_integer_chunks_values(list(chunks.values())):
        raise ValueError("Invalid 'chunks' values")
    # - Create dictionary for each variable
    new_chunks = {}
    for var in list(ds.data_vars):
        new_chunks[var] = {dim: chunks[dim] for dim in ds[var].dims}
    chunks = new_chunks
    return chunks


def check_chunks(ds, chunks, default_chunks=None):
    """
    Check chunks validity.

    The default_chunks argument is used only if chunks="auto".
    Chunk values -1 and None are equivalent to the entire dimension length !!!


    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset.
    chunks : (None, str, dict)
        Chunks of the dataset.
        If chunks = None, returns current dataset chunks.
        If chunks = "auto", it uses default_chunks.
        If chunks is a dictionary, two type of dictionaries are allowed:

            - 'per variable' chunks format: {<var>: {<dim>: <chunk_value>}}
            - 'per dimension' chunks format:  {<dims>: "auto"}

        If 'per dimension' chunks format is specified, it is converted to
        'per variable' chunks format.

    default_chunks : (None, dict), optional
        The defaults chunks to use to define chunks when chunks="auto".
        If default_chunks=None, it returns a dictionary of format: {<dims>: "auto"}
        The default is None.

    Returns
    -------
    chunks : dict
       Returns a 'per variable' chunks dictionary with format: {<var>: {<dim>: <chunk_value>}}.

    """
    # Input type checks
    chunks = _checks_chunks_type(chunks)
    if not isinstance(ds, xr.Dataset):
        raise TypeError("'ds' must be an xarray Dataset.")

    # If chunks = None, keep current chunks
    if isinstance(chunks, type(None)):
        chunks = get_dataset_chunks(ds)

    # If chunks = "auto", exploit default_chunks
    if isinstance(chunks, str):
        default_chunks = _check_default_chunks(default_chunks, ds)
        chunks = _use_default_chunks(ds, default_chunks)

    # Check chunks dictionary
    chunks = _check_chunks_dict(chunks=chunks, ds=ds)

    # Sanitize chunks (-1 and None to dimension length)
    chunks = sanitize_chunks_dict(chunks, ds)

    return chunks


def sanitize_chunks_dict(chunks_dict, ds):
    """Sanitize chunks dictionary.

    Change chunk value '-1' or None to length of the dataset dimension.
    Rechunk and zarr do not currently support -1 specification used by dask and xarray.
    """
    dict_dims = dict(ds.dims)
    for var in chunks_dict.keys():
        if chunks_dict[var] is not None:
            for k, v in chunks_dict[var].items():
                if v == -1 or isinstance(v, type(None)):
                    chunks_dict[var][k] = dict_dims[k]
    return chunks_dict


# -----------------------------------------------------------------------------.
#########################
#### Rechunker wrapper ##
#########################


def _check_zarr_store(target_store, force):
    """Check the Zarr target_store do not exist already."""
    if os.path.exists(target_store):
        if force:
            shutil.rmtree(target_store)
        else:
            raise ValueError(
                f"A zarr store already exists at {target_store}. If you want to overwrite, specify force=True"
            )


def rechunk_dataset(ds, chunks, target_store, temp_store, max_mem="1GB", force=False):
    """
    Rechunk on disk a xarray Dataset read lazily from a zarr store.

    Parameters
    ----------
    ds : xarray.Dataset
        A Dataset opened with open_zarr().
    chunks : dict
        Custom chunks of the new Dataset.
        If not specified for each Dataset variable, implicitly assumed.
    target_store : str
        Filepath of the zarr store where to save the new Dataset.
    temp_store : str
        Filepath of a zarr store where to save temporary data.
        This store is removed at the end of the rechunking operation.
    max_mem : str, optional
        The amount of memory (in bytes) that workers are allowed to use.
        The default is '1GB'.

    Returns
    -------
    None.

    """
    # TODO
    # - Add compressors options
    # compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    # options = dict(compressor=compressor)
    # rechunk(..., target_options=options)
    ##------------------------------------------------------------------------.

    # Check zarr stores
    _check_zarr_store(target_store, force=force)
    _check_zarr_store(temp_store, force=True)  # remove if exists

    # Check chunks
    target_chunks = check_chunks(ds=ds, chunks=chunks, default_chunks=None)

    # Plan rechunking
    r = rechunk(
        ds,
        target_chunks=target_chunks,
        max_mem=max_mem,
        target_store=target_store,
        temp_store=temp_store,
    )

    # Execute rechunking
    with ProgressBar():
        r.execute()

    # Remove temporary store
    _check_zarr_store(temp_store, force=True)  # remove if exists
