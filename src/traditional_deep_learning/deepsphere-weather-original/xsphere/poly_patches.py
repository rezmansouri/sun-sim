#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:55:53 2021

@author: ghiggi
"""
import numpy as np
import matplotlib.patches as mpatches


def get_polygons_2D_coords(lon_bnds, lat_bnds):
    """Create a list of numpy [x y] array polygon vertex coordinates from CDO lon_bnds and lat_bnds matrices."""
    # Input: n_polygons x n_vertices
    # Output: list (for each polygon) of numpy_array [x, y] polygon vertex coordinates
    n_polygons = lon_bnds.shape[0]
    n_vertices = lon_bnds.shape[1]
    list_polygons_xy = list()
    for i in range(n_polygons):
        poly_corners = np.zeros((n_vertices, 2), np.float64)
        poly_corners[:, 0] = lon_bnds[i, :]
        poly_corners[:, 1] = lat_bnds[i, :]
        list_polygons_xy.append(poly_corners)
    return list_polygons_xy


def get_PolygonPatchesList_from_latlon_bnds(lon_bnds, lat_bnds, fill=True):
    """Create a list of Polygon mpatches from CDO lon_bnds and lat_bnds."""
    # Construct list of polygons
    l_polygons_xy = get_polygons_2D_coords(lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    l_Polygon_patch = [
        mpatches.Polygon(xy=p, closed=False, fill=fill) for p in l_polygons_xy
    ]
    return l_Polygon_patch


def get_PolygonPatchesList(l_polygons_xy, fill=True):
    """Create Polygon mpatches from a numpy [x y] array with polygon vertex coordinates."""
    # Construct list of mpatches.Polygon
    l_Polygon_patch = [
        mpatches.Polygon(xy=p, closed=False, fill=fill) for p in l_polygons_xy
    ]
    return l_Polygon_patch
