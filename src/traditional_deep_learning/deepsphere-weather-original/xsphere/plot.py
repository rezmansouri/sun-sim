#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:37:01 2022

@author: ghiggi
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from scipy.interpolate import griddata
from xsphere.checks import check_xy, check_mesh_area_exist, check_mesh_exist
from xsphere.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
    import_matplotlib_pyplot,
    _process_cmap_cbar_kwargs,
    get_axis,
    _add_colorbar,
    label_from_attrs,
)
from xarray.plot import contour as xr_contour
from xarray.plot import contourf as xr_contourf
from xarray.plot import pcolormesh as xr_pcolormesh


# http://xarray.pydata.org/en/stable/generated/xarray.plot.FacetGrid.html
# https://github.com/pydata/xarray/blob/master/xarray/plot/plot.py


def contour(
    darray,
    x="lon",
    y="lat",
    transform=None,
    # Facetgrids arguments
    figsize=None,
    size=None,
    aspect=None,
    ax=None,
    row=None,
    col=None,
    col_wrap=None,
    subplot_kws=None,
    # Line option
    plot_type="contour",
    add_contour=True,
    linewidths=None,
    linestyles=None,
    antialiased=None,
    # Contour labels
    add_contour_labels=True,
    add_contour_labels_interactively=False,
    contour_labels_colors="black",
    contour_labels_fontsize="smaller",
    contour_labels_inline=True,
    contour_labels_inline_spacing=5,
    contour_labels_format="%1.3f",
    # Colors option
    alpha=1,
    colors=None,
    levels=None,
    cmap=None,
    norm=None,
    center=None,
    vmin=None,
    vmax=None,
    robust=False,
    extend="both",
    # Colorbar options
    add_colorbar=None,
    cbar_ax=None,
    cbar_kwargs=None,
    # Axis options
    add_labels=True,
    **kwargs
):
    """
    Contourf plotting method for unstructured mesh.

    The DataArray must have the attribute 'nodes'.

    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional, unless creating faceted plots.
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axis on which to plot this figure. By default, use the current axis.
        Mutually exclusive with ``size`` and ``figsize``.
    row : string, optional
        If passed, make row faceted plots on this dimension name
    col : string, optional
        If passed, make column faceted plots on this dimension name
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots
    add_contour : bool, optional
        Wheter to add contour lines. The default is True.
        Set to False is useful to plot just contour labels on top of a contourf plot.
    plot_type : str, optional
        Whether to use the "contour" or "tricontour" function.
        The default is "contour".
    linewidths : float, optional
        The line width of the contour lines.
        If a number, all levels will be plotted with this linewidth.
        If a sequence, the levels in ascending order will be plotted with the linewidths in the order specified.
        If None, this falls back to rcParams["lines.linewidth"] (default: 1.5).
        The default is None --> rcParams["lines.linewidth"]
    linestyles : str, optional
        If linestyles is None, the default is 'solid' unless the lines are monochrome.
        In that case, negative contours will take their linestyle from
        rcParams["contour.negative_linestyle"] (default: 'dashed') setting.
    antialiased: bool, optional
        Enable antialiasing, overriding the defaults.
        The default is taken from rcParams["lines.antialiased"].
    contour_labels_fontsize : string or float, optional
        Size in points or relative size e.g., 'smaller', 'x-large'.
    contour_labels_colors : color-spec, optional
        If None, the color of each label matches the color of the corresponding contour.
        If one string color, e.g., colors = 'r' or colors = 'red', all contour labels will be plotted in this color.
        If a tuple of matplotlib color args (string, float, rgb, etc),
        different contour labels will be plotted in different colors in the order specified.
    contour_labels_inline : bool, optional
        If True the underlying contour is removed where the label is placed. Default is True.
    contour_labels_inline_spacing : float, optional
        Space in pixels to leave on each side of contour label when placing inline. Defaults to 5.
        This spacing will be exact for contour labels at locations where the contour is straight,
        less so for labels on curved contours.
    contour_labels_fmt : string or dict, optional
        A format string for the label. Default is '%1.3f'
        Alternatively, this can be a dictionary matching contour levels with arbitrary strings
        to use for each contour level (i.e., fmt[level]=string),
        It can also be any callable, such as a Formatter instance (i.e. `"{:.0f} ".format`),
        that returns a string when called with a numeric contour level.
    contour_labels_manual : bool or iterable, optional
        If True, contour labels will be placed manually using mouse clicks.
        Click the first button near a contour to add a label,
        click the second button (or potentially both mouse buttons at once) to finish adding labels.
        The third button can be used to remove the last label added, but only if labels are not inline.
        Alternatively, the keyboard can be used to select label locations
        (enter to end label placement, delete or backspace act like the third mouse button,
         and any other key will select a label location).
        manual can also be an iterable object of x,y tuples.
        Contour labels will be created as if mouse is clicked at each x,y positions.
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    cmap : matplotlib colormap name or object, optional
        The mapping from data values to color space.
        If not provided, this will be either be ``viridis``
        (if the function infers a sequential dataset) or
        ``RdBu_r`` (if the function infers a diverging dataset).
        Is mutually exclusive with the color argument.
        When `Seaborn` is installed, ``cmap`` may also be a `seaborn`
        color palette.
        If ``cmap`` is a seaborn color palette, ``levels`` must not be specified.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    norm : ``matplotlib.colors.Normalize`` instance, optional
        If the ``norm`` has vmin or vmax specified, the corresponding kwarg
        must be None.
    colors : discrete colors to plot, optional
        A single color or a list of colors.
        Is mutually exclusive with cmap argument.
        Specification of ``levels`` argument is not mandatory.
    alpha : float, default: 1
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    extend : {"neither", "both", "min", "max"}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, extend is inferred from vmin, vmax and the data limits.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots. Only used
        for 2D and FacetGrid plots.
    add_colorbar : bool, optional
        Adds colorbar to axis
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar.
    add_labels : bool, optional
        Use xarray metadata to label axes
    **kwargs : optional
        Additional arguments to mpl.collections.PatchCollection
    Returns
    -------
    artist :
        The same type of primitive artist that the wrapped matplotlib
        function returns
    """
    ##------------------------------------------------------------------------.
    # Check is a DataArray
    if not isinstance(darray, xr.DataArray):
        raise TypeError("Provide a DataArray to xsphere._plot()")
    # Check plot_type
    if not isinstance(plot_type, str):
        raise TypeError(
            "'plot_type' must be a string: either 'contour' or 'tricontour'"
        )
    if plot_type not in ["contour", "tricontour"]:
        raise NotImplementedError(
            "'plot_type' accept only 'contour' or 'tricontour' options."
        )
    # Check ax
    if ax is None and row is None and col is None:
        raise ValueError("'ax' must be specified when not plotting a FacetGrids.")
    # Check transform
    if transform is None:
        transform = ccrs.PlateCarree()
    # Check x and y are coords of the xarray object
    check_xy(darray, x=x, y=y)
    ##------------------------------------------------------------------------.
    # Handle facetgrids first
    if row or col:
        if subplot_kws is None:
            print(
                "Tip: If you want to plot a map, you need to specify a projection \
                   using the argument subplot_kws={'projection': cartopy.crs.Robinson()}"
            )
        allargs = locals().copy()
        del allargs["darray"]
        allargs.update(allargs.pop("kwargs"))
        return _easy_facetgrid(data=darray, plotfunc=contour, **allargs)

    ##------------------------------------------------------------------------.
    # Initialize plot
    plt = import_matplotlib_pyplot()

    ##------------------------------------------------------------------------.
    # Pass the data as a masked ndarray too
    masked_arr = darray.to_masked_array(copy=False)

    ##------------------------------------------------------------------------.
    # Retrieve colormap and colorbar args
    cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
        xr_contour,
        masked_arr.data,
        **locals(),
        _is_facetgrid=kwargs.pop("_is_facetgrid", False)
    )
    ##------------------------------------------------------------------------.
    # If colors == 'a single color', matplotlib draws dashed negative contours.
    # We lose this feature if we pass cmap and not colors
    if isinstance(colors, str):
        cmap_params["cmap"] = None
    ##------------------------------------------------------------------------.
    # Define axis type
    if subplot_kws is None:
        subplot_kws = dict()
        ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
    ##------------------------------------------------------------------------.
    # Retrieve nodes coordinates
    lons = darray[x].values
    lats = darray[y].values
    ##------------------------------------------------------------------------.
    # Plot contour
    if plot_type == "tricontour":
        primitive = ax.tricontour(
            lons,
            lats,
            masked_arr.data,
            transform=transform,
            # Color options
            vmin=cmap_params["vmin"],
            vmax=cmap_params["vmax"],
            cmap=cmap_params["cmap"],
            norm=cmap_params["norm"],
            extend=cmap_params["extend"],
            levels=cmap_params["levels"],
            colors=colors,
            alpha=alpha,
            # Line options
            linewidths=linewidths,
            linestyles=linestyles,
            antialiased=antialiased,
            # Other args
            **kwargs
        )
    ##------------------------------------------------------------------------.
    # Plot with contour
    if plot_type == "contour":
        lons_new = np.linspace(-180, 180, 360 * 2)
        lats_new = np.linspace(-90, 90, 180 * 2)
        lons_2d, lats_2d = np.meshgrid(lons_new, lats_new)
        data_new = griddata(
            (lons, lats), masked_arr.data, (lons_2d, lats_2d), method="linear"
        )
        # Add a new longitude band at 360. equals to 0.
        data_new, lons_new = add_cyclic_point(data_new, coord=lons_new)
        # Plot contourf
        primitive = ax.contour(
            lons_new,
            lats_new,
            data_new,
            transform=transform,
            # Color options
            vmin=cmap_params["vmin"],
            vmax=cmap_params["vmax"],
            cmap=cmap_params["cmap"],
            norm=cmap_params["norm"],
            extend=cmap_params["extend"],
            levels=cmap_params["levels"],
            alpha=alpha,
            # Line options
            linewidths=linewidths,
            linestyles=linestyles,
            antialiased=antialiased,
            **kwargs
        )
    # Set global axis
    ax.set_global()
    ##------------------------------------------------------------------------.
    # Make the contours line invisible.
    if not add_contour:
        plt.setp(primitive.collections, visible=False)
    ##------------------------------------------------------------------------.
    # Add contour labels
    if add_contour_labels:
        ax.clabel(
            primitive,
            colors=contour_labels_colors,
            fontsize=contour_labels_fontsize,
            manual=add_contour_labels_interactively,
            inline=contour_labels_inline,
            inline_spacing=contour_labels_inline_spacing,
            fmt=contour_labels_format,
        )

    # Set global
    ax.set_global()
    ##------------------------------------------------------------------------.
    # Add labels
    if add_labels:
        ax.set_title(darray._title_for_slice())

    ##------------------------------------------------------------------------.
    # Add colorbar
    if add_colorbar:
        if "label" not in cbar_kwargs and add_labels:
            cbar_kwargs["label"] = label_from_attrs(darray)
        cbar = _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)
    else:
        # Inform the user about keywords which aren't used
        if cbar_ax is not None or cbar_kwargs:
            raise ValueError(
                "cbar_ax and cbar_kwargs can't be used with add_colorbar=False."
            )

    ##------------------------------------------------------------------------.
    return primitive


def contourf(
    darray,
    x="lon",
    y="lat",
    transform=None,
    # Facetgrids arguments
    figsize=None,
    size=None,
    aspect=None,
    ax=None,
    row=None,
    col=None,
    col_wrap=None,
    subplot_kws=None,
    # Colors option
    plot_type="contourf",
    antialiased=True,
    alpha=1,
    colors=None,
    levels=None,
    cmap=None,
    norm=None,
    center=None,
    vmin=None,
    vmax=None,
    robust=False,
    extend="both",
    # Colorbar options
    add_colorbar=None,
    cbar_ax=None,
    cbar_kwargs=None,
    # Axis options
    add_labels=True,
    **kwargs
):
    """
    Contourf plotting method for unstructured mesh.

    The DataArray must have the attribute 'nodes'.

    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional, unless creating faceted plots.
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axis on which to plot this figure. By default, use the current axis.
        Mutually exclusive with ``size`` and ``figsize``.
    row : string, optional
        If passed, make row faceted plots on this dimension name
    col : string, optional
        If passed, make column faceted plots on this dimension name
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots
    add_colorbar : bool, optional
        Adds colorbar to axis
    add_labels : bool, optional
        Use xarray metadata to label axes
    antialiased: bool, optional
        Enable antialiasing, overriding the defaults. For filled contours, the default is True.
    plot_type : str, optional
        Whether to use the "contourf" or "tricontourf" function.
        The default is "contour".
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    cmap : matplotlib colormap name or object, optional
        The mapping from data values to color space.
        If not provided, this will be either be ``viridis``
        (if the function infers a sequential dataset) or
        ``RdBu_r`` (if the function infers a diverging dataset).
        Is mutually exclusive with the color argument.
        When `Seaborn` is installed, ``cmap`` may also be a `seaborn`
        color palette.
        If ``cmap`` is a seaborn color palette, ``levels`` must not be specified.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    norm : ``matplotlib.colors.Normalize`` instance, optional
        If the ``norm`` has vmin or vmax specified, the corresponding kwarg
        must be None.
    colors : discrete colors to plot, optional
        A single color or a list of colors.
        Is mutually exclusive with cmap argument.
        Specification of ``levels`` argument is not mandatory.
    alpha : float, default: 1
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    extend : {"neither", "both", "min", "max"}, optional
        Determines the contourf-coloring of values that are outside the levels range and
        wheter to draw arrows extending the colorbar beyond its limits.
        If 'neither' (the default), values outside the levels range are not colored.
        If 'min', 'max' or 'both', color the values below, above or below and above the levels range.
        Values below min(levels) and above max(levels) are mapped to the under/over
        values of the Colormap.
        Note that most colormaps do not have dedicated colors for these by default,
        so that the over and under values are the edge values of the colormap.
        You may want to set these values explicitly using Colormap.set_under and Colormap.set_over.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots. Only used
        for 2D and FacetGrid plots.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar.
    **kwargs : optional
        Additional arguments to mpl.collections.PatchCollection
    Returns
    -------
    artist :
        The same type of primitive artist that the wrapped matplotlib
        function returns
    """
    # Check is a DataArray
    if not isinstance(darray, xr.DataArray):
        raise TypeError("Provide a DataArray to xsphere._plot()")
    # Checks plot_type
    if not isinstance(plot_type, str):
        raise TypeError(
            "'plot_type' must be a string: either 'contourf' or 'tricontourf'"
        )
    if plot_type not in ["contourf", "tricontourf"]:
        raise NotImplementedError(
            "'plot_type' accept only 'contourf' or 'tricontourf' options."
        )
    # Check ax
    if ax is None and row is None and col is None:
        raise ValueError("'ax' must be specified when not plotting a FacetGrids.")
    # Check transform
    if transform is None:
        transform = ccrs.PlateCarree()
    # Check x and y are coords of the xarray object
    check_xy(darray, x=x, y=y)
    ##------------------------------------------------------------------------.
    # Handle facetgrids first
    if row or col:
        if subplot_kws is None:
            print(
                "Tip: If you want to plot a map, you need to specify the projection \
                   using the argument subplot_kws={'projection': cartopy.crs.Robinson()}"
            )
        allargs = locals().copy()
        del allargs["darray"]
        allargs.update(allargs.pop("kwargs"))
        return _easy_facetgrid(data=darray, plotfunc=contourf, **allargs)

    ##------------------------------------------------------------------------.
    # Initialize plot
    plt = import_matplotlib_pyplot()

    ##------------------------------------------------------------------------.
    # Pass the data as a masked ndarray too
    masked_arr = darray.to_masked_array(copy=False)

    ##------------------------------------------------------------------------.
    # Retrieve colormap and colorbar args
    cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
        xr_contourf,
        masked_arr.data,
        **locals(),
        _is_facetgrid=kwargs.pop("_is_facetgrid", False)
    )
    ##------------------------------------------------------------------------.
    # If colors == 'a single color', matplotlib draws dashed negative contours.
    # We lose this feature if we pass cmap and not colors
    if isinstance(colors, str):
        cmap_params["cmap"] = None
        kwargs["colors"] = colors

    ##------------------------------------------------------------------------.
    # Define axis type
    if subplot_kws is None:
        subplot_kws = dict()
        ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
    ##------------------------------------------------------------------------.
    # Retrieve nodes coordinates
    lons = darray[x].values
    lats = darray[y].values
    # Plot with tricontourf
    if plot_type == "tricontourf":
        primitive = ax.tricontourf(
            lons,
            lats,
            masked_arr.data,
            transform=transform,
            # Color options
            vmin=cmap_params["vmin"],
            vmax=cmap_params["vmax"],
            cmap=cmap_params["cmap"],
            norm=cmap_params["norm"],
            extend=cmap_params["extend"],
            levels=cmap_params["levels"],
            alpha=alpha,
            antialiased=antialiased,
            **kwargs
        )
    # Plot with contourf
    if plot_type == "contourf":
        lons_new = np.linspace(-180, 180, 360 * 2)
        lats_new = np.linspace(-90, 90, 180 * 2)
        lons_2d, lats_2d = np.meshgrid(lons_new, lats_new)
        data_new = griddata(
            (lons, lats), masked_arr.data, (lons_2d, lats_2d), method="linear"
        )
        # Add a new longitude band at 360. equals to 0.
        data_new, lons_new = add_cyclic_point(data_new, coord=lons_new)
        # Plot contourf
        primitive = ax.contourf(
            lons_new,
            lats_new,
            data_new,
            transform=transform,
            # Color options
            vmin=cmap_params["vmin"],
            vmax=cmap_params["vmax"],
            cmap=cmap_params["cmap"],
            norm=cmap_params["norm"],
            extend=cmap_params["extend"],
            levels=cmap_params["levels"],
            alpha=alpha,
            antialiased=antialiased,
            **kwargs
        )
    # Set global axis
    ax.set_global()
    ##------------------------------------------------------------------------.
    # Add labels
    if add_labels:
        ax.set_title(darray._title_for_slice())

    ##------------------------------------------------------------------------.
    # Add colorbar
    if add_colorbar:
        if "label" not in cbar_kwargs and add_labels:
            cbar_kwargs["label"] = label_from_attrs(darray)
        cbar = _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)
    else:
        # Inform the user about keywords which aren't used
        if cbar_ax is not None or cbar_kwargs:
            raise ValueError(
                "cbar_ax and cbar_kwargs can't be used with add_colorbar=False."
            )

    ##------------------------------------------------------------------------.
    return primitive


def plot(
    darray,
    transform=None,
    # Facetgrids arguments
    figsize=None,
    size=None,
    aspect=None,
    ax=None,
    row=None,
    col=None,
    col_wrap=None,
    subplot_kws=None,
    # Polygon border option
    edgecolors="white",
    linewidths=0.1,
    antialiased=True,
    # Colors option
    colors=None,
    levels=None,
    cmap=None,
    norm=None,
    center=None,
    vmin=None,
    vmax=None,
    robust=False,
    extend="both",
    # Colorbar options
    add_colorbar=None,
    cbar_ax=None,
    cbar_kwargs=None,
    # Axis options
    add_labels=True,
    **kwargs
):
    """
    Plot the unstructured mesh using mpl.PatchCollection.

    The DataArray must have the attribute 'mesh' containing a mpl PolyPatch list.

    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional, unless creating faceted plots.
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axis on which to plot this figure. By default, use the current axis.
        Mutually exclusive with ``size`` and ``figsize``.
    row : string, optional
        If passed, make row faceted plots on this dimension name
    col : string, optional
        If passed, make column faceted plots on this dimension name
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots
    add_colorbar : bool, optional
        Adds colorbar to axis
    add_labels : bool, optional
        Use xarray metadata to label axes
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    cmap : matplotlib colormap name or object, optional
        The mapping from data values to color space.
        If not provided, this will be either be ``viridis``
        (if the function infers a sequential dataset) or
        ``RdBu_r`` (if the function infers a diverging dataset).
        Is mutually exclusive with the color argument.
        When `Seaborn` is installed, ``cmap`` may also be a `seaborn`
        color palette.
        If ``cmap`` is seaborn color palette, ``levels`` must also be specified.
    norm : ``matplotlib.colors.Normalize`` instance, optional
        If the ``norm`` has vmin or vmax specified, the corresponding kwarg
        must be None.
    colors : discrete colors to plot, optional
        A single color or a list of colors.
        Is mutually exclusive with cmap argument.
        Specification of ``levels`` argument is mandatory.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    extend : {"neither", "both", "min", "max"}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, extend is inferred from vmin, vmax and the data limits.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots. Only used
        for 2D and FacetGrid plots.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar.
    **kwargs : optional
        Additional arguments to mpl.collections.PatchCollection
    Returns
    -------
    artist :
        The same type of primitive artist that the wrapped matplotlib
        function returns
    """
    # Check is a DataArray
    if not isinstance(darray, xr.DataArray):
        raise TypeError("Provide a DataArray to xsphere._plot()")
    # Check ax
    if ax is None and row is None and col is None:
        raise ValueError("'ax' must be specified when not plotting a FacetGrids.")
    # Check transform
    if transform is None:
        transform = ccrs.Geodetic()
    # Check mesh is available
    check_mesh_exist(darray)
    ##------------------------------------------------------------------------.
    # Handle facetgrids first
    if row or col:
        if subplot_kws is None:
            print(
                "Tip: If you want to plot a map, you need to specify the projection \
                   using the argument subplot_kws={'projection': cartopy.crs.Robinson()}"
            )
        allargs = locals().copy()
        del allargs["darray"]
        allargs.update(allargs.pop("kwargs"))
        return _easy_facetgrid(data=darray, plotfunc=plot, **allargs)
    ##------------------------------------------------------------------------.
    # Initialize plot
    plt = import_matplotlib_pyplot()

    ##------------------------------------------------------------------------.
    # Pass the data as a masked ndarray too
    masked_arr = darray.to_masked_array(copy=False)

    ##------------------------------------------------------------------------.
    # Retrieve colormap and colorbar args
    cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
        xr_pcolormesh,
        masked_arr.data,
        **locals(),
        _is_facetgrid=kwargs.pop("_is_facetgrid", False)
    )
    ##------------------------------------------------------------------------.
    # Define axis type
    if subplot_kws is None:
        subplot_kws = dict()
        ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
    ##------------------------------------------------------------------------.
    # Create Polygon Patch Collection
    patch_list = darray["mesh"].values.tolist()
    Polygon_Collection = mpl.collections.PatchCollection(
        patch_list,
        transform=transform,
        array=masked_arr,
        # Polygon border
        edgecolors=edgecolors,
        linewidths=linewidths,
        antialiaseds=antialiased,
        # Color options
        cmap=cmap_params["cmap"],
        clim=(cmap_params["vmin"], cmap_params["vmax"]),
        norm=cmap_params["norm"],
        **kwargs
    )
    # Plot polygons
    ax.set_global()
    primitive = ax.add_collection(Polygon_Collection)

    ##------------------------------------------------------------------------.
    # Add labels
    if add_labels:
        ax.set_title(darray._title_for_slice())

    ##------------------------------------------------------------------------.
    # Add colorbar
    if add_colorbar:
        if "label" not in cbar_kwargs and add_labels:
            cbar_kwargs["label"] = label_from_attrs(darray)
        cbar = _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)
    else:
        # Inform the user about keywords which aren't used
        if cbar_ax is not None or cbar_kwargs:
            raise ValueError(
                "cbar_ax and cbar_kwargs can't be used with add_colorbar=False."
            )

    ##------------------------------------------------------------------------.
    return primitive


def plot_mesh(
    darray,
    ax,
    transform=None,
    add_background=True,
    antialiaseds=True,
    facecolors="none",
    edgecolors="black",
    linewidths=0.5,
    alpha=0.8,
    **kwargs
):
    """Plot the unstructured mesh.

    The DataArray must have the coordinate 'mesh' containing a mpl PolyPatch list.
    """
    # Check ax
    if ax is None:
        raise ValueError("'ax' must be specified when not plotting a FacetGrids.")
    # Check mesh is available
    check_mesh_exist(darray)
    # Check transform
    if transform is None:
        transform = ccrs.Geodetic()
    ##------------------------------------------------------------------------.
    # Retrieve mesh
    patch_list = darray["mesh"].values.tolist()
    # Create PatchCollection
    Polygon_Collection = mpl.collections.PatchCollection(
        patch_list,
        transform=transform,
        antialiaseds=antialiaseds,
        facecolors=facecolors,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths,
        **kwargs
    )
    # Plot the background
    ax.set_global()
    if add_background:
        ax.stock_img()
    # Plot the mesh
    primitive = ax.add_collection(Polygon_Collection)
    return primitive


def plot_mesh_order(
    darray,
    ax,
    transform=None,
    # Polygon border option
    edgecolors="white",
    linewidths=0.1,
    antialiased=True,
    # Colors option
    colors=None,
    levels=None,
    cmap=None,
    norm=None,
    center=None,
    vmin=None,
    vmax=None,
    robust=False,
    extend="neither",
    # Colorbar options
    add_colorbar=True,
    cbar_ax=None,
    cbar_kwargs=None,
    # Axis options
    add_labels=True,
    **kwargs
):
    """Plot the unstructured mesh order.

    The DataArray must have the coordinate 'mesh' containing a mpl PolyPatch list.
    """
    # Check mesh is available
    check_mesh_exist(darray)
    da = darray
    # Select 1 index in all dimensions (except node ...)
    dims = list(da.dims)
    node_dim = list(da["mesh"].dims)
    other_dims = np.array(dims)[np.isin(dims, node_dim, invert=True)].tolist()
    for dim in other_dims:
        da = da.isel({dim: 0})
    # Replace values with node order ...
    da.values = np.array(range(len(da["mesh"].values)))
    # Specify colorbar title
    if cbar_kwargs is None and add_colorbar:
        cbar_kwargs = {"label": "Mesh order ID"}
    # Plot mesh order
    primitive = da.sphere.plot(
        ax=ax,
        transform=transform,
        # Polygon border option
        edgecolors=edgecolors,
        linewidths=linewidths,
        antialiased=antialiased,
        # Colors option
        colors=colors,
        levels=levels,
        cmap=cmap,
        norm=norm,
        center=center,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        extend=extend,
        # Colorbar options
        add_colorbar=add_colorbar,
        cbar_ax=cbar_ax,
        cbar_kwargs=cbar_kwargs,
        # Axis options
        add_labels=add_labels,
        **kwargs
    )
    return primitive


def plot_mesh_area(
    darray,
    ax,
    transform=None,
    mesh_area_coord="area",
    # Polygon border option
    edgecolors="white",
    linewidths=0.1,
    antialiased=True,
    # Colors option
    colors=None,
    levels=None,
    cmap=None,
    norm=None,
    center=None,
    vmin=None,
    vmax=None,
    robust=False,
    extend="both",
    # Colorbar options
    add_colorbar=True,
    cbar_ax=None,
    cbar_kwargs=None,
    # Axis options
    add_labels=True,
    **kwargs
):
    """Plot the unstructured mesh area.

    The DataArray must have the coordinate 'mesh' containing a mpl PolyPatch list.
    """
    # Check mesh is available
    check_mesh_area_exist(darray, mesh_area_coord=mesh_area_coord)
    da = darray
    # Select 1 index in all dimensions (except node ...)
    dims = list(da.dims)
    node_dim = list(da[mesh_area_coord].dims)
    other_dims = np.array(dims)[np.isin(dims, node_dim, invert=True)].tolist()
    for dim in other_dims:
        da = da.isel({dim: 0})
    # Replace values with node order ...
    da.values = da[mesh_area_coord].values
    # Specify colorbar title
    if cbar_kwargs is None and add_colorbar:
        cbar_kwargs = {"label": "Area"}
    # Plot mesh order
    primitive = da.sphere.plot(
        ax=ax,
        transform=transform,
        # Polygon border option
        edgecolors=edgecolors,
        linewidths=linewidths,
        antialiased=antialiased,
        # Colors option
        colors=colors,
        levels=levels,
        cmap=cmap,
        norm=norm,
        center=center,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        extend=extend,
        # Colorbar options
        add_colorbar=add_colorbar,
        cbar_ax=cbar_ax,
        cbar_kwargs=cbar_kwargs,
        # Axis options
        add_labels=add_labels,
        **kwargs
    )
    return primitive


def plot_nodes(darray, ax, x="lon", y="lat", c="orange", add_background=True, **kwargs):
    # Check x and y are coords of the xarray object
    check_xy(darray, x=x, y=y)
    # Retrieve nodes coordinates
    lons = darray[x].values
    lats = darray[y].values
    # Add background
    if add_background:
        ax.stock_img()
    # Plot nodes
    primitive = ax.scatter(lons, lats, s=0.5, c=c, **kwargs)
    return primitive
