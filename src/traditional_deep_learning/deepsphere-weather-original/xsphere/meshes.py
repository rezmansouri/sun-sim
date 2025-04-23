#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:30:00 2022

@author: ghiggi
"""

"""Mesh generation tools"""
import numpy as np
import healpy as hp
from scipy.spatial import SphericalVoronoi
from xsphere.coordinates import xyz2lonlat, lonlat2xyz

# TODO: Refactor duplicated code !!!


def SphericalVoronoiMesh(lon, lat):
    """
    Infer the mesh of a spherical sampling from the mesh node centers provided in 2D geographic coordinates.

    Parameters
    ----------
    lon : numpy.ndarray
        Array of longitude coordinates (in degree units).
    lat : numpy.ndarray
        Array of latitude coordinates (in degree units).

    Returns
    -------
    list_polygons_lonlat : list
        List of numpy.ndarray with the polygon mesh vertices for each graph node.
    area : np.ndarray
        Numpy array with mesh area (in km²).
    """
    # Convert to geocentric coordinates
    radius = 6371.0e6  # radius = 1 can also be used
    x, y, z = lonlat2xyz(lon, lat, radius=radius)
    coords = np.column_stack((x, y, z))
    # Apply Spherical Voronoi tesselation
    sv = SphericalVoronoi(coords, radius=radius, center=[0, 0, 0])
    ##-------------------------------------------------------------------------.
    # SphericalVoronoi object methods
    # - sv.vertices : Vertex coords
    # - sv.regions : Vertex ID of each polygon
    # - sv.sort_vertices_of_regions() : sort indices of vertices to be clockwise ordered
    # - sv.calculate_areas() : compute the area of the spherical polygons
    ##-------------------------------------------------------------------------.
    # Sort vertices indexes to be clockwise ordered
    sv.sort_vertices_of_regions()
    # Retrieve area
    area = sv.calculate_areas()
    ##-------------------------------------------------------------------------.
    # Retrieve list of polygons coordinates arrays
    list_polygons_lonlat = []
    for region in sv.regions:
        tmp_xyz = sv.vertices[region]
        tmp_lon, tmp_lat = xyz2lonlat(
            tmp_xyz[:, 0], tmp_xyz[:, 1], tmp_xyz[:, 2], radius=radius
        )
        list_polygons_lonlat.append(np.column_stack((tmp_lon, tmp_lat)))
    ##-------------------------------------------------------------------------.
    return list_polygons_lonlat, area


# def SphericalVoronoiMesh_from_lonlat_coords(lon, lat):
#     """
#     Infer the mesh of a spherical sampling from the mesh node centers provided in 2D geographic coordinates.

#     Parameters
#     ----------
#     lon : numpy.ndarray
#         Array of longitude coordinates (in degree units).
#     lat : numpy.ndarray
#         Array of latitude coordinates (in degree units).

#     Returns
#     -------
#     list_polygons_lonlat : list
#         List of numpy.ndarray with the polygon mesh vertices for each graph node.

#     """
#     # Convert to geocentric coordinates
#     radius = 6371.0e6 # radius = 1 can also be used
#     x, y, z = lonlat2xyz(lon, lat, radius=radius)
#     coords = np.column_stack((x,y,z))
#     # Apply Spherical Voronoi tesselation
#     sv = SphericalVoronoi(coords,
#                           radius=radius,
#                           center=[0, 0, 0])
#     ##-------------------------------------------------------------------------.
#     # SphericalVoronoi object methods
#     # - sv.vertices : Vertex coords
#     # - sv.regions : Vertex ID of each polygon
#     # - sv.sort_vertices_of_regions() : sort indices of vertices to be clockwise ordered
#     # - sv.calculate_areas() : compute the area of the spherical polygons
#     ##-------------------------------------------------------------------------.
#     # Sort vertices indexes to be clockwise ordered
#     sv.sort_vertices_of_regions()
#     ##-------------------------------------------------------------------------.
#     # Retrieve list of polygons coordinates arrays
#     list_polygons_lonlat = []
#     for region in sv.regions:
#         tmp_xyz = sv.vertices[region]
#         tmp_lon, tmp_lat = xyz2lonlat(tmp_xyz[:,0],tmp_xyz[:,1],tmp_xyz[:,2], radius=radius)
#         list_polygons_lonlat.append(np.column_stack((tmp_lon, tmp_lat)))
#     ##-------------------------------------------------------------------------.
#     return list_polygons_lonlat


def SphericalVoronoiMesh_from_pygsp(graph):
    """
    Compute the mesh of a pygsp spherical graph using Spherical Voronoi.

    Parameters
    ----------
    graph : pgysp.graphs.nngraphs.sphere*
        pgysp graph of a spherical sampling.

    Returns
    -------
    list_polygons_lonlat : List
         List of numpy.ndarray with the polygon mesh vertices for each graph node.

    """
    radius = 1
    graph.set_coordinates("sphere", dim=3)
    sv = SphericalVoronoi(graph.coords, radius=radius, center=[0, 0, 0])
    sv.sort_vertices_of_regions()
    ##-------------------------------------------------------------------------.
    # Retrieve list of polygons coordinates arrays
    list_polygons_lonlat = []
    for region in sv.regions:
        tmp_xyz = sv.vertices[region]
        tmp_lon, tmp_lat = xyz2lonlat(
            tmp_xyz[:, 0], tmp_xyz[:, 1], tmp_xyz[:, 2], radius=radius
        )
        list_polygons_lonlat.append(np.column_stack((tmp_lon, tmp_lat)))
    ##-------------------------------------------------------------------------.
    return list_polygons_lonlat


def SphericalVoronoiMeshArea_from_pygsp(graph):
    """
    Compute the mesh of a pygsp spherical graph using Spherical Voronoi.

    Parameters
    ----------
    graph : pgysp.graphs.nngraphs.sphere*
        pgysp graph of a spherical sampling.

    Returns
    -------
    area : np.ndarray
         Numpy array with mesh area (in km²).

    """
    # Retrieve lon and lat coordinates
    graph.set_coordinates("sphere", dim=2)
    lat = np.rad2deg(graph.coords[:, 1])
    lon = np.rad2deg(graph.coords[:, 0])
    # Ensure lon is between -180 and 180
    lon[lon > 180] = lon[lon > 180] - 360
    # Convert to x,y,z geocentric coordinates
    radius = 6371  # km
    x, y, z = lonlat2xyz(lon, lat, radius=radius)
    coords = np.column_stack((x, y, z))
    # Apply Spherical Voronoi tesselation
    sv = SphericalVoronoi(coords, radius=radius, center=[0, 0, 0])
    area = sv.calculate_areas()
    return area


def HealpixMesh_from_pygsp(graph, step=16):
    """
    Compute the original quadrilateral polygons mesh of a pygsp SphereHealpix graph.

    Parameters
    ----------
    graph : pgysp.graphs.nngraphs.sphere*
        pgysp graph of a spherical sampling.
    step: int
        Govern accuracy of healpix mesh

    Returns
    -------
    list_polygons_lonlat : List
         List of numpy.ndarray with the polygon mesh vertices for each graph node.

    """
    # Retrieve HEALPix true vertices (quadrilateral polygons).
    radius = 1
    npix = graph.n_vertices
    nside = np.sqrt(npix / 12)
    vertices = hp.boundaries(nside, range(npix), nest=graph.nest, step=step)
    list_polygons_lonlat = []
    for tmp_xyz in vertices:
        tmp_lon, tmp_lat = xyz2lonlat(tmp_xyz[0], tmp_xyz[1], tmp_xyz[2], radius=radius)
        list_polygons_lonlat.append(np.column_stack((tmp_lon, tmp_lat)))
    ##-----------------------------------------------------------------------.
    return list_polygons_lonlat
