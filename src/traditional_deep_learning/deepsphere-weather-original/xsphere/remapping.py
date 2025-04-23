#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:31:34 2022

@author: ghiggi
"""
""" pygsp - xarray - CDO remapping wrappers"""
import os
import tempfile
import numpy as np
import xarray as xr
from xsphere.meshes import SphericalVoronoiMesh_from_pygsp, HealpixMesh_from_pygsp
from xsphere.cdo import (
    check_interp_method,
    check_normalization,
    cdo_genweights,
    cdo_remapping,
    write_cdo_grid,
)


def get_lat_lon_bnds(list_poly_lonlat):
    """
    Reshape a list of polygons in lat_bnds, lon_bnds array.

    Outputs arrays format:
    - Each row represent a polygon
    - Each column represent a polygon vertex coordinate

    The polygon with the largest number of vertices defines the number of columns of the matrices
    For polygons with less vertices, the last vertex coordinates is repeated.

    Parameters
    ----------
    list_poly_lonlat : list
        List of numpy.ndarray with the polygon mesh vertices for each graph node.

    Returns
    -------
    lon_bnds : numpy.ndarray
        Array with the longitude vertices of each polygon.
    lat_bnds : numpy.ndarray
         Array with the latitude vertices of each polygon.

    """
    # Retrieve max number of vertex
    n_max_vertex = 0
    for p_lonlat in list_poly_lonlat:
        if len(p_lonlat) > n_max_vertex:
            n_max_vertex = len(p_lonlat)
    # Create polygon x vertices coordinates arrays
    n_poly = len(list_poly_lonlat)
    lat_bnds = np.empty(shape=(n_poly, n_max_vertex)) * np.nan
    lon_bnds = np.empty(shape=(n_poly, n_max_vertex)) * np.nan
    for i, p_lonlat in enumerate(list_poly_lonlat):
        tmp_lons = p_lonlat[:, 0]
        tmp_lats = p_lonlat[:, 1]
        if (
            len(tmp_lats) < n_max_vertex
        ):  # Repeat the last vertex to have n_max_vertex values
            for _ in range(n_max_vertex - len(tmp_lats)):
                tmp_lons = np.append(tmp_lons, tmp_lons[-1])
                tmp_lats = np.append(tmp_lats, tmp_lats[-1])
        lat_bnds[i, :] = tmp_lats.tolist()
        lon_bnds[i, :] = tmp_lons.tolist()
    return lon_bnds, lat_bnds


def _write_dummy_1D_nc(graph, fpath=None):
    """Create a dummy netCDF for CDO based on pygsp graph."""
    # The dummy netCDF is required by CDO to compute the interpolation weights.
    ##-------------------------------------------------------------------------.
    # Create dummy filepath
    if fpath is None:
        fpath = tempfile.NamedTemporaryFile(
            prefix="dummy_1D_netcdf_", suffix=".nc"
        ).name
    ##-------------------------------------------------------------------------.
    # Create dummy netCDF
    n = graph.n_vertices
    data = np.arange(0, n)
    da = xr.DataArray(
        data=data[np.newaxis],
        dims=["time", "nodes"],
        coords={"time": np.datetime64("2005-02-25")[np.newaxis]},
        name="dummy_var",
    )
    # da.coords["lat"] = ('nodes', data)
    # da.coords["lon"] = ('nodes', data)
    ds = da.to_dataset()
    ds.to_netcdf(fpath)
    return fpath


def pygsp_to_CDO_grid(graph, CDO_grid_fpath, rounding=4):
    """
    Define CDO grid based on pygsp Spherical graph.

    Parameters
    ----------
    graph : pygsp.graph
        pygsp spherical graph.
    CDO_grid_fpath : str
        Filepath where to save the CDO grid.
    rounding: int
        Rounding decimal digits of lat/lon coordinates to reduce total number of vertices

    Returns
    -------
    None.

    """
    ## TODO Check is pygsp graph
    ##-------------------------------------------------------------------------.
    # Retrieve graph nodes
    lon_center = graph.signals["lon"] * 180 / np.pi
    lat_center = graph.signals["lat"] * 180 / np.pi
    # Enforce longitude between -180 and 180
    lon_center[lon_center > 180] = lon_center[lon_center > 180] - 360
    # Consider it as cell centers and infer it vertex
    list_polygons_lonlat = SphericalVoronoiMesh_from_pygsp(graph)  # PolygonArrayList
    # Approximate to 2 decimal digits to reduce the number of mesh vertices (if > 8)
    # max_n_vertices = max([len(x) for x in list_polygons_lonlat])
    # if max_n_vertices > 8:
    #     list_polygons_lonlat1 = []
    #     for i, p in enumerate(list_polygons_lonlat):
    #         list_polygons_lonlat1.append(np.unique(np.round(p,rounding), axis = 0))
    #     list_polygons_lonlat = list_polygons_lonlat1

    # Reformat polygon vertices array to have all same number of vertices
    lon_vertices, lat_vertices = get_lat_lon_bnds(list_polygons_lonlat)
    # Write down the ECMF mesh into CDO grid format
    write_cdo_grid(
        fpath=CDO_grid_fpath,
        xvals=lon_center,
        yvals=lat_center,
        xbounds=lon_vertices,
        ybounds=lat_vertices,
    )
    return


def pygsp_Healpix_to_CDO_grid(graph, CDO_grid_fpath):
    """
    Define CDO grid of a pygsp SphericalHealpix graph.

    Parameters
    ----------
    graph : pygsp.graph
        pygsp spherical graph.
    CDO_grid_fpath : str
        Filepath where to save the CDO grid.

    Returns
    -------
    None.

    """
    ## TODO Check is pygsp graph
    ##-------------------------------------------------------------------------.
    # Retrieve graph nodes
    lon_center = graph.signals["lon"] * 180 / np.pi
    lat_center = graph.signals["lat"] * 180 / np.pi
    # Enforce longitude between -180 and 180
    lon_center[lon_center > 180] = lon_center[lon_center > 180] - 360
    # Retrieve original Healpix quadrilateral polygons
    list_polygons_lonlat = HealpixMesh_from_pygsp(graph)  # PolygonArrayList
    # Reformat polygon vertices array to have all same number of vertices
    lon_vertices, lat_vertices = get_lat_lon_bnds(list_polygons_lonlat)
    # Write down the ECMF mesh into CDO grid format
    write_cdo_grid(
        fpath=CDO_grid_fpath,
        xvals=lon_center,
        yvals=lat_center,
        xbounds=lon_vertices,
        ybounds=lat_vertices,
    )
    return


def compute_interpolation_weights(
    src_graph,
    dst_graph,
    method="conservative",
    normalization="fracarea",
    weights_fpath=None,
    src_CDO_grid_fpath=None,
    dst_CDO_grid_fpath=None,
    recreate_CDO_grids=False,
    return_weights=True,
    n_threads=1,
    verbose=False,
):
    """
    Compute interpolation weights between two pygsp spherical samplings.

    Parameters
    ----------
    src_graph : pygsp.graph
        Source spherical graph.
    dst_graph : pygsp.graph
        Destination spherical graph.
    method : str, optional
        Interpolation/remapping method. The default is "conservative".
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
    weights_fpath : str, optional
        Optional filepath where to save the weights netCDF4. The default is None.
        If None, the weights are not saved on disk.
    src_CDO_grid_fpath : str, optional
        Filepath where to save the CDO grid for the source spherical grid. The default is None.
        If None, the CDO grid is not saved on disk.
    dst_CDO_grid_fpath : str, optional
        Filepath where to save the CDO grid for the destination spherical grid. The default is None.
        If None, the CDO grid is not saved on disk.
    recreate_CDO_grids : bool, optional
        Wheter to redefine the CDO grids if src_CDO_grid_fpath or dst_CDO_grid_fpath are provided.
        The default is False.
    n_threads : int, optional
        Number of threads to compute the interpolation weights. The default is 1.
    return_weights : bool, optional
        Wheter to return the interpolation weights. The default is True.

    Returns
    -------
    ds : xarray.Dataset
        Xarray Dataset with the interpolation weights.

    """
    # Check arguments
    check_interp_method(method)
    check_normalization(normalization)
    # Check boolean arguments
    if not isinstance(recreate_CDO_grids, bool):
        raise TypeError("'recreate_CDO_grids' must be either True or False")
    if not isinstance(return_weights, bool):
        raise TypeError("'return_weights' must be either True or False")
    ##-------------------------------------------------------------------------.
    # Create temporary fpath if required
    FLAG_tmp_src_CDO_grid_fpath = False
    FLAG_tmp_dst_CDO_grid_fpath = False
    FLAG_tmp_weights_fpath = False
    if src_CDO_grid_fpath is None:
        FLAG_tmp_src_CDO_grid_fpath = True
        src_CDO_grid_fpath = tempfile.NamedTemporaryFile(prefix="src_CDO_grid_").name
    if dst_CDO_grid_fpath is None:
        FLAG_tmp_dst_CDO_grid_fpath = True
        dst_CDO_grid_fpath = tempfile.NamedTemporaryFile(prefix="dst_CDO_grid_").name
    if weights_fpath is None:
        FLAG_tmp_weights_fpath = True
        weights_fpath = tempfile.NamedTemporaryFile(
            prefix="CDO_weights_", suffix=".nc"
        ).name
    ##------------------------------------------------------------------------.
    # Checks if the directory exists
    if not os.path.exists(os.path.dirname(weights_fpath)):
        raise ValueError(
            "The directory where to store the interpolation weights do not exists."
        )
    if not os.path.exists(os.path.dirname(src_CDO_grid_fpath)):
        raise ValueError(
            "The directory where to store the CDO (input) grid do not exists."
        )
    if not os.path.exists(os.path.dirname(dst_CDO_grid_fpath)):
        raise ValueError(
            "The directory where to store the CDO (output) grid do not exists."
        )
    ##-------------------------------------------------------------------------.
    # Define CDO grids based on pygsp graph if required
    if recreate_CDO_grids or FLAG_tmp_src_CDO_grid_fpath:
        pygsp_to_CDO_grid(src_graph, src_CDO_grid_fpath)
    if recreate_CDO_grids or FLAG_tmp_dst_CDO_grid_fpath:
        pygsp_to_CDO_grid(dst_graph, dst_CDO_grid_fpath)
    ##-------------------------------------------------------------------------.
    # Create a dummy input file for CDO
    src_fpath = _write_dummy_1D_nc(src_graph)
    ##-------------------------------------------------------------------------.
    # Compute interpolation weights
    cdo_genweights(
        method,
        src_CDO_grid_fpath=src_CDO_grid_fpath,
        dst_CDO_grid_fpath=dst_CDO_grid_fpath,
        src_fpath=src_fpath,
        weights_fpath=weights_fpath,
        normalization=normalization,
        n_threads=n_threads,
        verbose=verbose,
    )
    ##-------------------------------------------------------------------------.
    # Load the weights if required
    if return_weights:
        ds = xr.open_dataset(weights_fpath)
    ##-------------------------------------------------------------------------.
    # Remove dummy files
    os.remove(src_fpath)
    if FLAG_tmp_weights_fpath:
        os.remove(weights_fpath)
    if FLAG_tmp_src_CDO_grid_fpath:
        os.remove(src_CDO_grid_fpath)
    if FLAG_tmp_dst_CDO_grid_fpath:
        os.remove(dst_CDO_grid_fpath)
    ##-------------------------------------------------------------------------.
    if return_weights:
        return ds
    else:
        return


##----------------------------------------------------------------------------.
def remap_dataset(
    src_ds,
    src_graph=None,
    src_CDO_grid_fpath=None,
    dst_graph=None,
    dst_CDO_grid_fpath=None,
    method="conservative",
    normalization="fracarea",
    remapped_ds_fpath=None,
    return_remapped_ds=True,
    compression_level=1,
    n_threads=1,
):
    """
    Remap an xarray Dataset using CDO.

    Either provide pygsp graphs (i.e. for unstructured Spherical grids)
    or the filepath of CDO grids defining source and destination grids.

    Parameters
    ----------
    src_ds : xarray.Dataset
        xarray Dataset to remap.
    src_graph : pygsp.graph
        Source spherical graph.
    dst_graph : pygsp.graph
        Destination spherical graph.
    src_CDO_grid_fpath : str, optional
        Filepath of the CDO grid for the source spherical grid.
    dst_CDO_grid_fpath : str, optional
        Filepath of the CDO grid for the destination spherical grid.
    method : str, optional
        Interpolation/remapping method. The default is "conservative".
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
    remapped_ds_fpath : str, optional
        Optional filepath where to save the remapped Dataset in netCDF4 format.
        The default is None. If None, the remapped Dataset is not saved on disk.
    return_remapped_ds : bool, optional
        Wheter to return the remapped Dataset. The default is True.
    compression_level : int, optional
        Compression level of the NetCDF4 file when saving it to disk.
        The default is 1. Valid values between 0 and 9. 0 means no compression.
    n_threads : int, optional
        Number of threads to use when performing remapping. The default is 1.

    Returns
    -------
    ds_remapped : xarray.Dataset
        The remapped dataset.

    """
    ##-------------------------------------------------------------------------.
    # Check input arguments
    check_interp_method(method)
    check_normalization(normalization)
    # Check boolean arguments
    if not isinstance(return_remapped_ds, bool):
        raise TypeError("'return_remapped_ds' must be either True or False")
    ##-------------------------------------------------------------------------.
    # Initialize flags
    FLAG_src_graph_provided = False
    FLAG_dst_graph_provided = False
    FLAG_tmp_src_CDO_grid_fpath = False
    FLAG_tmp_dst_CDO_grid_fpath = False
    FLAG_tmp_remapped_ds_fpath = False
    ##-------------------------------------------------------------------------.
    # Check <src/dst>_graph and <src_dst>_CDO_grid_fpath not boths None
    if (src_graph is None) and (src_CDO_grid_fpath is None):
        raise ValueError("Please provide pygsp 'src_graph' or 'src_CDO_grid_fpath'")
    if (dst_graph is None) and (dst_CDO_grid_fpath is None):
        raise ValueError("Please provide pygsp 'src_graph' or 'dst_CDO_grid_fpath'")
    # If <src/dst>_graph and <src_dst>_CDO_grid_fpath boths provided, just use src_graph
    if (src_graph is not None) and (src_CDO_grid_fpath is not None):
        print(
            "Warning: Both 'src_graph' and 'src_CDO_grid_fpath' provided. Discarding 'src_CDO_grid_fpath'"
        )
        src_CDO_grid_fpath = tempfile.NamedTemporaryFile(prefix="src_CDO_grid_").name
        FLAG_tmp_src_CDO_grid_fpath = True
    if (dst_graph is not None) and (dst_CDO_grid_fpath is not None):
        print(
            "Warning: Both 'dst_graph' and 'dst_CDO_grid_fpath' provided. Discarding 'dst_CDO_grid_fpath'"
        )
        dst_CDO_grid_fpath = tempfile.NamedTemporaryFile(prefix="dst_CDO_grid_").name
        FLAG_tmp_dst_CDO_grid_fpath = True
    ##------------------------------------------------------------------------.
    # Check that src_graph match dimensions of ds
    if src_graph.n_vertices not in list(src_ds.dims.values()):
        raise ValueError(
            "'src_ds' doest not have a dimension length equal to 'src_graph.n_vertices'."
        )
    ##------------------------------------------------------------------------.
    # Check provided CDO grids exists
    if src_CDO_grid_fpath is not None:
        if not os.path.exists(src_CDO_grid_fpath):
            raise ValueError(
                "The specified 'src_CDO_grid_fpath' do not exists. Provide valid filepath"
            )
    if dst_CDO_grid_fpath is not None:
        if not os.path.exists(dst_CDO_grid_fpath):
            raise ValueError(
                "The specified 'dst_CDO_grid_fpath' do not exists. Provide valid filepath"
            )
    ##-------------------------------------------------------------------------.
    # Create temporary fpath if required
    if src_graph is not None:
        FLAG_src_graph_provided = True
    if dst_graph is not None:
        FLAG_dst_graph_provided = True
    if src_CDO_grid_fpath is None:
        FLAG_tmp_src_CDO_grid_fpath = True
        src_CDO_grid_fpath = tempfile.NamedTemporaryFile(prefix="src_CDO_grid_").name
    if dst_CDO_grid_fpath is None:
        FLAG_tmp_dst_CDO_grid_fpath = True
        dst_CDO_grid_fpath = tempfile.NamedTemporaryFile(prefix="dst_CDO_grid_").name
    if remapped_ds_fpath is None:
        FLAG_tmp_remapped_ds_fpath = True
        remapped_ds_fpath = tempfile.NamedTemporaryFile(
            prefix="tmp_remapped_netcdf_", suffix=".nc"
        ).name
    ##-------------------------------------------------------------------------.
    # Checks if the directory exists
    if not os.path.exists(os.path.dirname(remapped_ds_fpath)):
        raise ValueError(
            "The directory where to store the remapped Dataset do not exists."
        )
    ##-------------------------------------------------------------------------.
    # Define CDO grids based on pygsp graph if required
    if FLAG_src_graph_provided:
        pygsp_to_CDO_grid(src_graph, src_CDO_grid_fpath)
    if FLAG_dst_graph_provided:
        pygsp_to_CDO_grid(dst_graph, dst_CDO_grid_fpath)
    ##-------------------------------------------------------------------------.
    # Save the source (input) dataset to disk temporary
    tmp_src_ds_fpath = tempfile.NamedTemporaryFile(
        prefix="tmp_input_netcdf_", suffix=".nc"
    ).name
    src_ds.to_netcdf(tmp_src_ds_fpath)
    ##-------------------------------------------------------------------------.
    # Compute interpolation weights
    cdo_remapping(
        method=method,
        src_CDO_grid_fpath=src_CDO_grid_fpath,
        dst_CDO_grid_fpath=dst_CDO_grid_fpath,
        src_fpaths=tmp_src_ds_fpath,
        dst_fpaths=remapped_ds_fpath,
        precompute_weights=False,
        normalization=normalization,
        compression_level=compression_level,
        n_threads=n_threads,
    )
    ##-------------------------------------------------------------------------.
    # Load the weights if required
    if return_remapped_ds:
        ds_remapped = xr.open_dataset(remapped_ds_fpath)
        # When dealing with unstructured data (i.e.pygsp Spherical graph)
        if src_graph is not None:
            ds_remapped = ds_remapped.rename({"ncells": "nodes"})
    ##-------------------------------------------------------------------------.
    # Remove dummy files
    os.remove(tmp_src_ds_fpath)
    if FLAG_tmp_remapped_ds_fpath:
        os.remove(remapped_ds_fpath)  # dest ds
    if FLAG_tmp_src_CDO_grid_fpath:
        os.remove(src_CDO_grid_fpath)
    if FLAG_tmp_dst_CDO_grid_fpath:
        os.remove(dst_CDO_grid_fpath)
    ##-------------------------------------------------------------------------.
    if return_remapped_ds:
        return ds_remapped
    else:
        return


###---------------------------------------------------------------------------.
def compute_interpolation_weights_Healpix(
    src_graph,
    dst_graph,
    method="conservative",
    normalization="fracarea",
    weights_fpath=None,
    src_CDO_grid_fpath=None,
    dst_CDO_grid_fpath=None,
    recreate_CDO_grids=False,
    return_weights=True,
    n_threads=1,
    verbose=False,
):
    """
    Compute interpolation weights between two Healpix samplings.

    Use original Healpix Mesh instead of Spherical Voronoi.

    Parameters
    ----------
    src_graph : pygsp.graph
        Source spherical graph.
    dst_graph : pygsp.graph
        Destination spherical graph.
    method : str, optional
        Interpolation/remapping method. The default is "conservative".
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
    weights_fpath : str, optional
        Optional filepath where to save the weights netCDF4. The default is None.
        If None, the weights are not saved on disk.
    src_CDO_grid_fpath : str, optional
        Filepath where to save the CDO grid for the source spherical grid. The default is None.
        If None, the CDO grid is not saved on disk.
    dst_CDO_grid_fpath : str, optional
        Filepath where to save the CDO grid for the destination spherical grid. The default is None.
        If None, the CDO grid is not saved on disk.
    recreate_CDO_grids : bool, optional
        Wheter to redefine the CDO grids if src_CDO_grid_fpath or dst_CDO_grid_fpath are provided.
        The default is False.
    n_threads : int, optional
        Number of threads to compute the interpolation weights. The default is 1.
    return_weights : bool, optional
        Wheter to return the interpolation weights. The default is True.

    Returns
    -------
    ds : xarray.Dataset
        Xarray Dataset with the interpolation weights.

    """
    # Check arguments
    check_interp_method(method)
    check_normalization(normalization)
    # Check boolean arguments
    if not isinstance(recreate_CDO_grids, bool):
        raise TypeError("'recreate_CDO_grids' must be either True or False")
    if not isinstance(return_weights, bool):
        raise TypeError("'return_weights' must be either True or False")
    ##------------------------------------------------------------------------.
    # Create temporary fpath if required
    FLAG_tmp_src_CDO_grid_fpath = False
    FLAG_tmp_dst_CDO_grid_fpath = False
    FLAG_tmp_weights_fpath = False
    if src_CDO_grid_fpath is None:
        FLAG_tmp_src_CDO_grid_fpath = True
        src_CDO_grid_fpath = tempfile.NamedTemporaryFile(prefix="src_CDO_grid_").name
    if dst_CDO_grid_fpath is None:
        FLAG_tmp_dst_CDO_grid_fpath = True
        dst_CDO_grid_fpath = tempfile.NamedTemporaryFile(prefix="dst_CDO_grid_").name
    if weights_fpath is None:
        FLAG_tmp_weights_fpath = True
        weights_fpath = tempfile.NamedTemporaryFile(
            prefix="CDO_weights_", suffix=".nc"
        ).name
    ##------------------------------------------------------------------------.
    # Checks if the directory exists
    if not os.path.exists(os.path.dirname(weights_fpath)):
        raise ValueError(
            "The directory where to store the interpolation weights do not exists."
        )
    if not os.path.exists(os.path.dirname(src_CDO_grid_fpath)):
        raise ValueError(
            "The directory where to store the CDO (input) grid do not exists."
        )
    if not os.path.exists(os.path.dirname(dst_CDO_grid_fpath)):
        raise ValueError(
            "The directory where to store the CDO (output) grid do not exists."
        )
    ##------------------------------------------------------------------------.
    # Define CDO grids based on pygsp graph if required
    if recreate_CDO_grids or FLAG_tmp_src_CDO_grid_fpath:
        pygsp_Healpix_to_CDO_grid(src_graph, src_CDO_grid_fpath)
    if recreate_CDO_grids or FLAG_tmp_dst_CDO_grid_fpath:
        pygsp_Healpix_to_CDO_grid(dst_graph, dst_CDO_grid_fpath)
    ##------------------------------------------------------------------------.
    # Create a dummy input file for CDO
    src_fpath = _write_dummy_1D_nc(src_graph)
    ##------------------------------------------------------------------------.
    # Compute interpolation weights
    cdo_genweights(
        method,
        src_CDO_grid_fpath=src_CDO_grid_fpath,
        dst_CDO_grid_fpath=dst_CDO_grid_fpath,
        src_fpath=src_fpath,
        weights_fpath=weights_fpath,
        normalization=normalization,
        n_threads=n_threads,
        verbose=verbose,
    )
    ##------------------------------------------------------------------------.
    # Load the weights if required
    if return_weights:
        ds = xr.open_dataset(weights_fpath)
    ##------------------------------------------------------------------------.
    # Remove dummy files
    os.remove(src_fpath)
    if FLAG_tmp_weights_fpath:
        os.remove(weights_fpath)
    if FLAG_tmp_src_CDO_grid_fpath:
        os.remove(src_CDO_grid_fpath)
    if FLAG_tmp_dst_CDO_grid_fpath:
        os.remove(dst_CDO_grid_fpath)
    ##------------------------------------------------------------------------.
    if return_weights:
        return ds
    else:
        return


##----------------------------------------------------------------------------.
