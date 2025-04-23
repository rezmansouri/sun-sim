#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:47:44 2022

@author: ghiggi
"""

from xarray.plot.facetgrid import FacetGrid
from xarray.plot.utils import _process_cmap_cbar_kwargs
from xarray.plot import pcolormesh as xr_pcolormesh
from xarray.plot import contour as xr_contour
from xarray.plot import contourf as xr_contourf


def map_dataarray_unstructured(self, func, **kwargs):
    """
    Apply a plotting function to an unstructured grid subset of data.

    This is more convenient and less general than ``FacetGrid.map``

    Parameters
    ----------
    func : callable
        A plotting function
    kwargs
        Additional keyword arguments to func
    Returns
    -------
    self : FacetGrid object
    """
    if kwargs.get("cbar_ax", None) is not None:
        raise ValueError("cbar_ax not supported by FacetGrid.")
    ##------------------------------------------------------------------------.
    # Colorbar settings (exploit xarray defaults)
    if func.__name__ == "contour":
        xr_func = xr_contour
    if func.__name__ == "contourf":
        xr_func = xr_contourf
    if func.__name__ == "plot":
        xr_func = xr_pcolormesh
    cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
        xr_func, self.data.values, **kwargs
    )
    self._cmap_extend = cmap_params.get("extend")
    ##------------------------------------------------------------------------.
    # Order is important
    func_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in {"cmap", "colors", "cbar_kwargs", "levels"}
    }
    func_kwargs.update(cmap_params)
    func_kwargs.update({"add_colorbar": False, "add_labels": False})
    ##------------------------------------------------------------------------.
    # Plot
    for d, ax in zip(self.name_dicts.flat, self.axes.flat):
        # None is the sentinel value
        if d is not None:
            subset = self.data.loc[d]
            mappable = func(subset, ax=ax, **func_kwargs, _is_facetgrid=True)
            self._mappables.append(mappable)
    ##------------------------------------------------------------------------.
    xlabel = ""
    ylabel = ""
    self._finalize_grid(xlabel, ylabel)
    ##------------------------------------------------------------------------.
    # Add colorbars
    if kwargs.get("add_colorbar", True):
        self.add_colorbar(**cbar_kwargs)
    ##------------------------------------------------------------------------.
    return self


def _easy_facetgrid(
    data,
    plotfunc,
    x=None,
    y=None,
    row=None,
    col=None,
    col_wrap=None,
    sharex=True,
    sharey=True,
    aspect=None,
    size=None,
    subplot_kws=None,
    ax=None,
    figsize=None,
    **kwargs,
):
    """
    Call xarray.plot.FacetGrid from the plotting methods.

    kwargs are the arguments to the plotting method.
    """
    if ax is not None:
        raise ValueError("Can't use axes when making faceted plots.")
    if aspect is None:
        aspect = 1
    if size is None:
        size = 3
    elif figsize is not None:
        raise ValueError("cannot provide both `figsize` and `size` arguments")

    g = FacetGrid(
        data=data,
        col=col,
        row=row,
        col_wrap=col_wrap,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        aspect=aspect,
        size=size,
        subplot_kws=subplot_kws,
    )
    # Add map_dataarray_unstructured to FacetGrid
    g.map_dataarray_unstructured = map_dataarray_unstructured
    # Plot
    return g.map_dataarray_unstructured(g, plotfunc, **kwargs)
