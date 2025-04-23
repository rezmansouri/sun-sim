#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:36:35 2022

@author: ghiggi
"""
"""CDO remapping tools"""
import os
import tempfile
import subprocess
import numpy as np

###----------------------------------------------------------------------------.
# Utils function used to write CDO grid files
def arr2str(arr):
    """Convert numpy 1D array into single string of spaced numbers."""
    return "  ".join(map(str, list(arr)))


###----------------------------------------------------------------------------.
def write_cdo_grid(fpath, xvals, yvals, xbounds, ybounds):
    """
    Create the CDO Grid Description File of an unstructured grid.

    Parameters
    ----------
    fpath : str
        CDO Grid Description File name/path to write
    xvals : numpy.ndarray
        The longitude center of each grid cell.
    yvals : numpy.ndarray
       The latitude center of each grid cell.
    xbounds : numpy.ndarray
        The longitudes of the corners of each grid cell.
    ybounds : numpy.ndarray
        The latitudes of the corners of each grid cell.

    Returns
    -------
    None.

    """
    # Checks
    if not isinstance(yvals, np.ndarray):
        raise TypeError("Provide yvals as numpy.ndarray")
    if not isinstance(xvals, np.ndarray):
        raise TypeError("Provide xvals as numpy.ndarray")
    if not isinstance(xbounds, np.ndarray):
        raise TypeError("Provide xbounds as numpy.ndarray")
    if not isinstance(ybounds, np.ndarray):
        raise TypeError("Provide ybounds as numpy.ndarray")
    if len(yvals) != len(xvals):
        raise ValueError("xvals and yvals must have same size")
    if ybounds.shape[0] != xbounds.shape[0]:
        raise ValueError("xbounds and ybounds must have same shape")
    ##------------------------------------------------------------------------.
    # Retrieve number of patch and max number of vertex
    n_cells = len(yvals)
    nvertex = ybounds.shape[1]
    # Write down the gridType
    with open(fpath, "w") as txt_file:
        txt_file.write("gridtype  = unstructured \n")
        txt_file.write("gridsize  = %s \n" % (n_cells))
        txt_file.write("nvertex   = %s \n" % (nvertex))
        # Write xvals
        txt_file.write("xvals     = %s \n" % (arr2str(xvals)))
        # Write yvals
        txt_file.write("yvals     = %s \n" % (arr2str(yvals)))
        # Write xbounds
        txt_file.write("xbounds   = %s \n" % (arr2str(xbounds[0, :])))
        for line in xbounds[1:, :]:
            txt_file.write("            %s \n" % (arr2str(line)))
        # Write ybounds
        txt_file.write("ybounds   = %s \n" % (arr2str(ybounds[0, :])))
        for line in ybounds[1:, :]:
            txt_file.write("            %s \n" % (arr2str(line)))


def get_available_interp_methods():
    """Available interpolation methods."""
    methods = [
        "nearest_neighbors",
        "idw",
        "bilinear",
        "bicubic",
        "conservative",
        "conservative_SCRIP",
        "conservative2",
        "largest_area_fraction",
    ]
    return methods


def check_interp_method(method):
    """Check if interpolation method is valid."""
    if not isinstance(method, str):
        raise TypeError("Provide interpolation 'method' name as a string")
    if method not in get_available_interp_methods():
        raise ValueError(
            "Provide valid interpolation method. get_available_interp_methods()"
        )


def check_normalization(normalization):
    """Check normalization option for CDO conservative remapping."""
    if not isinstance(normalization, str):
        raise TypeError("Provide 'normalization' type as a string")
    if normalization not in ["fracarea", "destarea"]:
        raise ValueError("Normalization must be either 'fracarea' or 'destarea'")


def get_cdo_genweights_cmd(method):
    """Define available methods to generate interpolation weights in CDO."""
    d = {
        "nearest_neighbors": "gennn",
        "idw": "gendis",
        "bilinear": "genbil",
        "bicubic": "genbic",
        "conservative": "genycon",
        "conservative_SCRIP": "gencon",
        "conservative2": "genycon2",
        "largest_area_fraction": "genlaf",
    }
    return d[method]


def get_cdo_remap_cmd(method):
    """Define available interpolation methods in CDO."""
    # REMAPDIS - IDW using the 4 nearest neighbors
    d = {
        "nearest_neighbors": "remapnn",
        "idw": "remapdis",
        "bilinear": "remapbil",
        "bicubic": "remapbic",
        "conservative": "remapycon",
        "conservative_SCRIP": "remapcon",
        "conservative2": "remapycon2",
        "largest_area_fraction": "remaplaf",
    }
    return d[method]


def cdo_genweights(
    method,
    src_CDO_grid_fpath,
    dst_CDO_grid_fpath,
    src_fpath,
    weights_fpath,
    normalization="fracarea",
    n_threads=1,
    verbose=True,
):
    """
    Wrap around CDO gen* to compute interpolation weights.

    Parameters
    ----------
    method : str
        Interpolation method.
    src_CDO_grid_fpath : str
        File (path) specifying the grid structure of input data.
    dst_CDO_grid_fpath : str
        File (path) specifying the grid structure of output data.
    src_fpath : str
        Filepath of the input file
    weights_fpath : str
        Filepath of the CDO interpolation weights.
    normalization : str, optional
        Normalization option for conservative remapping.
        The default is 'fracarea'.
        Options:
        - fracarea uses the sum of the non-masked source cell intersected
          areas to normalize each target cell field value.
          Flux is not locally conserved.
        - destarea’ uses the total target cell area to normalize each target
          cell field value.
          Local flux conservation is ensured, but unreasonable flux values
          may result [i.e. in small patches].
    compression_level : int, optional
        Compression level of output netCDF4. Default 1. 0 for no compression.
    n_threads : int, optional
        Number of OpenMP threads to use within CDO. The default is 1.

    Returns
    -------
    None.

    """
    # TODO: to generalize to whatever case, src_CDO_grid_fpath and -setgrid faculative if real data ..
    ##------------------------------------------------------------------------.
    # Check input arguments
    check_interp_method(method)
    check_normalization(normalization)
    ##------------------------------------------------------------------------.
    # Check that the folder where to save the weights it exists
    if not os.path.exists(os.path.dirname(weights_fpath)):
        raise ValueError(
            "The directory where to store the interpolation weights do not exists"
        )
    ##------------------------------------------------------------------------.
    ## Define CDO options for interpolation (to be exported in the environment)
    # - cdo_grid_search_radius
    # - remap_extrapolate
    opt_CDO_environment = "".join(
        ["CDO_REMAP_NORM", "=", "'", normalization, "'" "; " "export CDO_REMAP_NORM; "]
    )
    ##------------------------------------------------------------------------.
    ## Define CDO OpenMP threads options
    if n_threads > 1:
        opt_CDO_parallelism = "--worker %s -P %s" % (n_threads, n_threads)
        # opt_CDO_parallelism = "-P %s" %(n_threads) # necessary for CDO < 1.9.8
    else:
        opt_CDO_parallelism = ""
    ##------------------------------------------------------------------------.
    ## Define output precision
    if method != "largest_area_fraction":
        output_precision = "-b 64"
    else:
        output_precision = ""
    ##------------------------------------------------------------------------.
    ## Compute weights
    cdo_genweights_command = get_cdo_genweights_cmd(method=method)
    # Define command
    command = "".join(
        [
            opt_CDO_environment,
            "cdo ",
            opt_CDO_parallelism,
            " ",
            output_precision,
            " ",
            cdo_genweights_command,
            ",",
            dst_CDO_grid_fpath,
            " ",
            "-setgrid,",
            src_CDO_grid_fpath,
            " ",
            src_fpath,
            " ",
            weights_fpath,
        ]
    )
    # Run command
    capture_output = not verbose
    flag_cmd = subprocess.run(command, shell=True, capture_output=capture_output)
    if flag_cmd.returncode != 0:
        raise ValueError(
            "An error with code {} occured during the computation of interpolation weights with CDO.".format(
                flag_cmd.returncode
            )
        )
    return


def cdo_remapping(
    method,
    src_CDO_grid_fpath,
    dst_CDO_grid_fpath,
    src_fpaths,
    dst_fpaths,
    precompute_weights=True,
    weights_fpath=None,
    normalization="fracarea",
    compression_level=1,
    n_threads=1,
):
    """
    Wrap around CDO to remap grib files to whatever unstructured grid.

    Parameters
    ----------
    method : str
        Interpolation method.
    src_CDO_grid_fpath : str
        File (path) specifying the grid structure of input data.
    dst_CDO_grid_fpath : str
        File (path) specifying the grid structure of output data.
    src_fpaths : list
        Filepaths list of input data to remap.
    dst_fpaths : list
        Filepaths list where to save remapped data.
    precompute_weights : bool, optional
        Whether to use or first precompute the interpolation weights.
        The default is True.
    weights_fpath : str, optional
        Filepath of the CDO interpolation weights.
        It is used only if precompute_weights is True.
        If not specified, it save the interpolation weights in a temporary
        folder which is deleted when processing ends.
    normalization : str, optional
        Normalization option for conservative remapping.
        The default is 'fracarea'.
        Options:
        - fracarea uses the sum of the non-masked source cell intersected
          areas to normalize each target cell field value.
          Flux is not locally conserved.
        - destarea’ uses the total target cell area to normalize each target
          cell field value.
          Local flux conservation is ensured, but unreasonable flux values
          may result [i.e. in small patches].
    compression_level : int, optional
        Compression level of output netCDF4. Default 1. 0 for no compression.
    n_threads : int, optional
        Number of OpenMP threads to use within CDO. The default is 1.

    Returns
    -------
    None.

    """
    # TODO: to generalize to whatever case, src_CDO_grid_fpath and -setgrid faculative if real data ...
    ##------------------------------------------------------------------------.
    # Check arguments
    check_interp_method(method)
    check_normalization(normalization)
    # Check boolean arguments
    if not isinstance(precompute_weights, bool):
        raise TypeError("'precompute_weights' must be either True or False")
    # Check src_fpaths, dst_fpaths
    # - Ensure are list
    if isinstance(src_fpaths, str):
        src_fpaths = [src_fpaths]
    if isinstance(dst_fpaths, str):
        dst_fpaths = [dst_fpaths]
    # - Check is list
    if not ((isinstance(dst_fpaths, list) or isinstance(src_fpaths, list))):
        raise TypeError("Provide 'src_fpaths' and 'dst_fpaths' as list (or str)")
    # - Check same length
    if len(src_fpaths) != len(dst_fpaths):
        raise ValueError("'src_fpaths' and 'dst_fpaths' must have same length.")
    ##-------------------------------------------------------------------------
    # Define temporary path for weights if weights are precomputed
    FLAG_temporary_weight = False
    if precompute_weights:
        FLAG_temporary_weight = False
        if weights_fpath is None:
            # Create temporary directory where to store interpolation weights
            FLAG_temporary_weight = True
            weights_fpath = tempfile.NamedTemporaryFile(
                prefix="CDO_weights_", suffix=".nc"
            ).name
        else:
            # Check that the folder where to save the weights it exists
            if not os.path.exists(os.path.dirname(weights_fpath)):
                raise ValueError(
                    "The directory where to store the interpolation weights do not exists"
                )
    ##------------------------------------------------------------------------.
    ## Define CDO options for interpolation (to export in the environment)
    # - cdo_grid_search_radius
    # - remap_extrapolate
    opt_CDO_environment = "".join(
        [
            "CDO_REMAP_NORM",
            "=",
            "'",
            normalization,
            "'",
            "; ",
            "export CDO_REMAP_NORM; ",
        ]
    )
    ##------------------------------------------------------------------------.
    ## Define CDO OpenMP threads options
    if n_threads > 1:
        opt_CDO_parallelism = "--worker %s -P %s" % (n_threads, n_threads)
        # opt_CDO_parallelism = "-P %s" %(n_threads) # necessary for CDO < 1.9.8
    else:
        opt_CDO_parallelism = ""
    ##------------------------------------------------------------------------.
    ## Define netCDF4 compression options
    if compression_level > 9 or compression_level < 1:
        opt_CDO_data_compression = "-z zip_%s" % (int(compression_level))
    else:
        opt_CDO_data_compression = ""
    ##------------------------------------------------------------------------.
    ## Define output precision
    if method != "largest_area_fraction":
        output_precision = "-b 64"
    else:
        # BUG in genlaf (reported) ... need to use remaplaf directly for the moment
        precompute_weights = False
        output_precision = ""
    ##------------------------------------------------------------------------.
    ## Precompute the weights (once) and then remap
    if precompute_weights:
        ##--------------------------------------------------------------------.
        # If weights are not yet pre-computed, compute it
        if not os.path.exists(weights_fpath):
            cdo_genweights_command = get_cdo_genweights_cmd(method=method)
            # Define command
            command = "".join(
                [
                    opt_CDO_environment,
                    "cdo ",
                    opt_CDO_parallelism,
                    " ",
                    output_precision,
                    " ",  # output precision
                    cdo_genweights_command,
                    ",",
                    dst_CDO_grid_fpath,
                    " ",
                    "-setgrid,",
                    src_CDO_grid_fpath,
                    " ",
                    src_fpaths[0],
                    " ",
                    weights_fpath,
                ]
            )
            # Run command
            flag_cmd = subprocess.run(command, shell=True, capture_output=False)
            if flag_cmd.returncode != 0:
                raise ValueError(
                    "An error occured during the computation of interpolation weights with CDO."
                )
        ##--------------------------------------------------------------------.
        # Remap all files
        for src_fpath, dst_fpath in zip(src_fpaths, dst_fpaths):
            # Define command
            command = "".join(
                [
                    opt_CDO_environment,
                    "cdo ",
                    opt_CDO_parallelism,
                    " ",
                    output_precision,
                    " ",
                    "-f nc4",
                    " ",  # output type: netcdf
                    opt_CDO_data_compression,
                    " ",
                    "remap,",
                    dst_CDO_grid_fpath,
                    ",",
                    weights_fpath,
                    " ",
                    "-setgrid,",
                    src_CDO_grid_fpath,
                    " ",
                    src_fpath,
                    " ",
                    dst_fpath,
                ]
            )
            # Run command
            flag_cmd = subprocess.run(command, shell=True, capture_output=False)
            if flag_cmd.returncode != 0:
                raise ValueError("An error occured during remapping data with CDO.")
    ##--------------------------------------------------------------------.
    ## Remap directly without precomputing the weights
    else:
        # Retrieve CDO command for direct interpolation
        remapping_command = get_cdo_remap_cmd(method=method)
        # Remap all files
        for src_fpath, dst_fpath in zip(src_fpaths, dst_fpaths):
            # Define command
            command = "".join(
                [
                    opt_CDO_environment,
                    "cdo ",
                    opt_CDO_parallelism,
                    " ",
                    output_precision,
                    " ",
                    "-f nc4",
                    " ",
                    opt_CDO_data_compression,
                    " ",
                    remapping_command,
                    ",",
                    dst_CDO_grid_fpath,
                    " ",
                    "-setgrid,",
                    src_CDO_grid_fpath,
                    " ",
                    src_fpath,
                    " ",
                    dst_fpath,
                ]
            )
            # Run command
            flag_cmd = subprocess.run(command, shell=True, capture_output=False)
            if flag_cmd.returncode != 0:
                raise ValueError("An error occured during remapping data with CDO.")
    ##-------------------------------------------------------------------------.
    if FLAG_temporary_weight:
        os.remove(weights_fpath)
    return


# -----------------------------------------------------------------------------.
