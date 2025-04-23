#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:27:44 2022

@author: ghiggi
"""
import shapely
import numpy as np
import xarray as xr
from xsphere.checks import check_xy, check_mesh, check_mesh_exist, check_node_dim
from xsphere.poly_patches import get_PolygonPatchesList
from xsphere.meshes import SphericalVoronoiMesh

# Refactor SphereDatasetAccesor ... too much repeated code


class Sphere_Base_Accessor:
    def __init__(self, xarray_obj):
        if not isinstance(xarray_obj, (xr.DataArray, xr.Dataset)):
            raise TypeError(
                "The 'sphere' accessor is available only for xr.Dataset and xr.DataArray."
            )
        self._obj = xarray_obj

    def add_nodes(self, lon, lat, node_dim="node"):
        """Add unstructured grid nodes."""
        # Check node_dim
        node_dim = check_node_dim(self._obj, node_dim=node_dim)
        # Add nodes coordinates
        self._obj = self._obj.assign_coords({"lon": (node_dim, lon)})
        self._obj = self._obj.assign_coords({"lat": (node_dim, lat)})
        return self._obj

    def add_nodes_from_pygsp(self, pygsp_graph, node_dim="node"):
        """Add nodes from the spherical pygsp graph."""
        # Retrieve lon and lat coordinates
        pygsp_graph.set_coordinates("sphere", dim=2)
        lon = np.rad2deg(pygsp_graph.coords[:, 0])
        lat = np.rad2deg(pygsp_graph.coords[:, 1])
        # Ensure lon is between -180 and 180
        lon[lon > 180] = lon[lon > 180] - 360
        return self.add_nodes(lon=lon, lat=lat, node_dim=node_dim)

    def add_mesh(self, mesh, node_dim="node"):
        """Add unstructured grid mesh."""
        # Check node_dim
        node_dim = check_node_dim(self._obj, node_dim=node_dim)
        # Check mesh
        mesh = check_mesh(mesh)
        # Add mesh as xarray coordinate
        self._obj = self._obj.assign_coords({"mesh": (node_dim, mesh)})
        return self._obj

    def add_mesh_area(self, area, node_dim="node"):
        """Add unstructured grid mesh area."""
        # Check node_dim
        node_dim = check_node_dim(self._obj, node_dim=node_dim)
        # Add mesh as xarray coordinate
        self._obj = self._obj.assign_coords({"area": (node_dim, area)})
        return self._obj

    def compute_mesh_area(self):
        """Compute the mesh area."""
        # TODO: Improve for computing on the sphere... now using shapely planar assumption
        # Scipy - Spherical? https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/spatial/_spherical_voronoi.py#L266
        ##---------------------------------------------------------------------.
        # Check the mesh exist
        check_mesh_exist(self._obj)
        # Retrieve mesh
        mesh = self._obj["mesh"].values
        # Retrieve node_dim
        node_dim = list(self._obj["mesh"].dims)[0]
        # Compute area (with shapely ... planar assumption)
        area = [shapely.geometry.Polygon(p.xy).area for p in mesh]
        return self.add_mesh_area(area=area, node_dim=node_dim)

    def add_SphericalVoronoiMesh(self, x="lon", y="lat", add_area=True):
        """Compute the Spherical Voronoi mesh using the node coordinates."""
        # Check x and y are coords of the xarray object
        check_xy(self._obj, x=x, y=y)
        # Retrieve node coordinates
        lon = self._obj[x].values
        lat = self._obj[y].values
        node_dim = list(self._obj[x].dims)[0]
        # Compute SphericalVoronoi Mesh
        list_polygons_lonlat, area = SphericalVoronoiMesh(lon=lon, lat=lat)
        mesh = get_PolygonPatchesList(list_polygons_lonlat)
        # Add mesh
        self.add_mesh(mesh=mesh, node_dim=node_dim)
        # Add mesh area
        if add_area:
            self.add_mesh_area(area=area, node_dim=node_dim)
        return self._obj

    def has_mesh(self):
        # Check mesh attached
        # TODO: check also format of mesh
        if "mesh" not in list(self._obj.coords.keys()):
            return False
        else:
            return True


@xr.register_dataarray_accessor("sphere")
class SphereDataArrayAccessor(Sphere_Base_Accessor):
    """xarray.sphere DataArray accessor."""

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def plot(self, *args, **kwargs):
        """Map unstructured grid values."""
        from .plot import plot

        p = plot(self._obj, *args, **kwargs)
        return p

    def contour(self, *args, **kwargs):
        """Contour of unstructured grid values."""
        from .plot import contour

        p = contour(self._obj, *args, **kwargs)
        return p

    def contourf(self, *args, **kwargs):
        """Contourf of unstructured grid values."""
        from .plot import contourf

        p = contourf(self._obj, *args, **kwargs)
        return p

    def plot_mesh(self, *args, **kwargs):
        """Plot the unstructured grid mesh structure."""
        from .plot import plot_mesh

        p = plot_mesh(self._obj, *args, **kwargs)
        return p

    def plot_mesh_order(self, *args, **kwargs):
        """Plot the unstructured grid mesh order."""
        from .plot import plot_mesh_order

        p = plot_mesh_order(self._obj, *args, **kwargs)
        return p

    def plot_mesh_area(self, *args, **kwargs):
        """Plot the unstructured grid mesh area."""
        from .plot import plot_mesh_area

        p = plot_mesh_area(self._obj, *args, **kwargs)
        return p

    def plot_nodes(self, *args, **kwargs):
        """Plot the unstructured grid nodes."""
        from .plot import plot_nodes

        p = plot_nodes(self._obj, *args, **kwargs)
        return p


@xr.register_dataset_accessor("sphere")
class SphereDatasetAccessor(Sphere_Base_Accessor):
    """xarray.sphere Dataset accessor."""

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def plot(self, col=None, row=None, *args, **kwargs):
        """Map unstructured grid values."""
        ds = self._obj
        # Check number of variable
        list_vars = list(ds.data_vars.keys())
        n_vars = len(list_vars)
        # If only 1 variable, treat as DataArray
        if n_vars == 1:
            return ds[list_vars[0]].sphere.plot(col=col, row=row, *args, **kwargs)
        # Otherwise
        if col is not None and row is not None:
            raise ValueError(
                "When plotting a Dataset, you must specify either 'row' or 'col'."
            )
        if col is None and row is None:
            raise NotImplementedError("Specify 'col' or 'row'.")
        # Squeeze the dataset (to drop dim with 1)
        ds = ds.squeeze()
        # Check remaining dimension
        if len(ds.dims) > 2:
            raise ValueError(
                "There must be just 1 dimension to facet (in addition to the 'node' dimension)."
            )
        # Convert to DataArray
        da = ds.to_array()
        if col is not None:
            p = da.sphere.plot(row="variable", col=col, *args, **kwargs)
            return p
        elif row is not None:
            p = da.sphere.plot(col="variable", row=row, *args, **kwargs)
            return p
        else:
            raise NotImplementedError("When 'col' and 'row' are both None (END).")

    def contour(self, col=None, row=None, *args, **kwargs):
        """Contour of unstructured grid values."""
        ds = self._obj
        # Check number of variable
        list_vars = list(ds.data_vars.keys())
        n_vars = len(list_vars)
        # If only 1 variable, treat as DataArray
        if n_vars == 1:
            return ds[list_vars[0]].sphere.contour(col=col, row=row, *args, **kwargs)
        # Otherwise
        if col is not None and row is not None:
            raise ValueError(
                "When contourting a Dataset, you must specify either 'row' or 'col'."
            )
        if col is None and row is None:
            raise NotImplementedError("Specify 'col' or 'row'.")
        # Squeeze the dataset (to drop dim with 1)
        ds = self._obj.squeeze()
        # Check remaining dimension
        if len(ds.dims) > 2:
            raise ValueError(
                "There must be just 1 dimension to facet (in addition to the 'node' dimension)."
            )
        # Convert to DataArray
        da = self._obj.to_array()
        if col is not None:
            p = da.sphere.contour(row="variable", col=col * args, **kwargs)
            return p
        elif row is not None:
            p = da.sphere.contour(col="variable", row=row, *args, **kwargs)
            return p
        else:
            raise NotImplementedError("When 'col' and 'row' are both None (END).")

    def contourf(self, col=None, row=None, *args, **kwargs):
        """Contourf of unstructured grid values."""
        ds = self._obj
        # Check number of variable
        list_vars = list(ds.data_vars.keys())
        n_vars = len(list_vars)
        # If only 1 variable, treat as DataArray
        if n_vars == 1:
            return ds[list_vars[0]].sphere.contourf(col=col, row=row, *args, **kwargs)
        # Otherwise
        if col is not None and row is not None:
            raise ValueError(
                "When contourfting a Dataset, you must specify either 'row' or 'col'."
            )
        if col is None and row is None:
            raise NotImplementedError("Specify 'col' or 'row'.")
        # Squeeze the dataset (to drop dim with 1)
        ds = self._obj.squeeze()
        # Check remaining dimension
        if len(ds.dims) > 2:
            raise ValueError(
                "There must be just 1 dimension to facet (in addition to the 'node' dimension)."
            )
        # Convert to DataArray
        da = self._obj.to_array()
        if col is not None:
            p = da.sphere.contourf(row="variable", col=col * args, **kwargs)
            return p
        elif row is not None:
            p = da.sphere.contourf(col="variable", row=row, *args, **kwargs)
            return p
        else:
            raise NotImplementedError("When 'col' and 'row' are both None (END).")

    def plot_mesh(self, *args, **kwargs):
        """Plot the unstructured grid mesh structure."""
        from .plot import plot_mesh

        da = self._obj[list(self._obj.data_vars.keys())[0]]
        p = plot_mesh(da, *args, **kwargs)
        return p

    def plot_mesh_order(self, *args, **kwargs):
        """Plot the unstructured grid mesh order."""
        from .plot import plot_mesh_order

        da = self._obj[list(self._obj.data_vars.keys())[0]]
        p = plot_mesh_order(da, *args, **kwargs)
        return p

    def plot_mesh_area(self, *args, **kwargs):
        """Plot the unstructured grid mesh area."""
        from .plot import plot_mesh_area

        da = self._obj[list(self._obj.data_vars.keys())[0]]
        p = plot_mesh_area(da, *args, **kwargs)
        return p

    def plot_nodes(self, *args, **kwargs):
        """Plot the unstructured grid nodes."""
        from .plot import plot_nodes

        da = self._obj[list(self._obj.data_vars.keys())[0]]
        p = plot_nodes(da, *args, **kwargs)
        return p
