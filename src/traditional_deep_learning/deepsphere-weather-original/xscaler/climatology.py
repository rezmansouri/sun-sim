#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:53:28 2022

@author: ghiggi
"""
import xarray as xr
import numpy as np
from .temporal_scalers import TemporalStandardScaler
from .checks import _check_is_fitted, _check_save_fpath


class Climatology:
    """Compute climatology."""

    def __init__(
        self,
        data,
        time_dim,
        time_groups=None,
        groupby_dims=None,
        variable_dim=None,
        mean=True,
        variability=True,
        reference_period=None,
        ds_climatology=None,
    ):
        # reference_period : numpy array or tuple of len 2 with datetime
        # ('1980-01-01T00:00','2010-12-31T23:00')
        # np.array(['1980-01-01T00:00','2010-12-31T23:00'], dtype='M8')
        ##--------------------------------------------------------------------.
        # Create a climatology object if ds_climatology is a xr.Dataset
        if isinstance(ds_climatology, xr.Dataset):
            self.fitted = True
            self.mean_ = ds_climatology["Mean"].to_dataset(dim="variable")
            self.variability_ = ds_climatology["Variability"].to_dataset(dim="variable")
            self.variable_dim = (
                ds_climatology.attrs["variable_dim"]
                if ds_climatology.attrs["variable_dim"] != "None"
                else None
            )
            self.aggregating_dims = (
                ds_climatology.attrs["aggregating_dims"]
                if ds_climatology.attrs["aggregating_dims"] != "None"
                else None
            )
            self.groupby_dims = (
                ds_climatology.attrs["groupby_dims"]
                if ds_climatology.attrs["groupby_dims"] != "None"
                else None
            )
            self.time_dim = ds_climatology.attrs["time_dim"]
            self.time_groups = eval(ds_climatology.attrs["time_groups"])
            self.time_groupby_factors = eval(
                ds_climatology.attrs["time_groupby_factors"]
            )
            self.time_groupby_name = ds_climatology.attrs["time_groupby_name"]
            # Ensure arguments are list
            if isinstance(self.aggregating_dims, str):
                self.aggregating_dims = [self.aggregating_dims]
            if isinstance(self.groupby_dims, str):
                self.groupby_dims = [self.groupby_dims]
        ##--------------------------------------------------------------------.
        # Initialize climatology object
        else:
            self.scaler = TemporalStandardScaler(
                data=data,
                time_dim=time_dim,
                time_groups=time_groups,
                variable_dim=variable_dim,
                groupby_dims=groupby_dims,
                reference_period=reference_period,
                center=mean,
                standardize=variability,
            )
            self.fitted = False

    ##------------------------------------------------------------------------.
    def compute(self):
        """Compute climatology mean and variability."""
        # Fit scaler
        self.scaler.fit()
        self.fitted = self.scaler.fitted
        # Retrieve mean and variability
        self.mean_ = self.scaler.mean_
        self.variability_ = self.scaler.std_
        # Extract time group dimensions
        self.time_dim = self.scaler.time_dim
        self.time_groups = self.scaler.time_groups
        self.time_groupby_factors = self.scaler.time_groupby_factors
        self.time_groupby_name = self.scaler.time_groupby_name
        time_groupby_dims = (
            self.scaler.time_groupby_dims
        )  # not self because not saved to disk
        # Extract other infos
        self.variable_dim = self.scaler.variable_dim
        self.aggregating_dims = self.scaler.aggregating_dims
        self.groupby_dims = self.scaler.groupby_dims

        # Add extended time group dimensions
        if self.mean_ is not None:
            for k in self.time_groups.keys():
                if k == "weekofyear":
                    k = "week"
                self.mean_[k] = time_groupby_dims[k]
                self.mean_ = self.mean_.set_coords(k)
        if self.variability_ is not None:
            for k in self.time_groups.keys():
                if k == "weekofyear":
                    k = "week"
                self.variability_[k] = time_groupby_dims[k]
                self.variability_ = self.variability_.set_coords(k)
        # Return DataArray if input data is dataarray
        if self.variable_dim is not None:
            if self.mean_ is not None:
                self.mean_ = self.mean_.to_array(self.variable_dim)
            if self.variability_ is not None:
                self.variability_ = self.variability_.to_array(self.variable_dim)

    @property
    def mean(self):
        if self.mean_ is not None:
            # Remove time groups
            ds = self.mean_.drop(self.time_groupby_name)
            ds = ds.swap_dims({self.time_groupby_name: "time_group"})
            return ds

    @property
    def variability(self):
        if self.variability_ is not None:
            # Remove time groups
            ds = self.variability_.drop(self.time_groupby_name)
            ds = ds.swap_dims({self.time_groupby_name: "time_group"})
            return ds

    ##------------------------------------------------------------------------.
    def save(self, fpath, force=False):
        """Save the Climatogy object to disk in netCDF format."""
        # Check
        _check_is_fitted(self)
        fpath = _check_save_fpath(fpath=fpath, force=force)
        ##---------------------------------------------------------------------.
        ## Create Climatology xarray Dataset (to save as netCDF)
        # - Reshape mean and variability into DataArray
        if self.mean_ is not None:
            mean_ = self.mean_.to_array()
            mean_.name = "Mean"
        if self.variability_ is not None:
            std_ = self.variability_.to_array()
            std_.name = "Variability"
        # - Pack data into a Dataset
        if self.mean_ is not None and self.variability_ is not None:
            ds_clim = xr.merge((mean_, std_))
        elif self.mean_ is not None:
            ds_clim = mean_.to_dataset()
        else:
            ds_clim = std_.to_dataset()
        # Add attributes
        ds_clim.attrs = {
            "aggregating_dims": self.aggregating_dims
            if self.aggregating_dims is not None
            else "None",
            "groupby_dims": self.groupby_dims
            if self.groupby_dims is not None
            else "None",
            "variable_dim": self.variable_dim
            if self.variable_dim is not None
            else "None",
            "time_dim": self.time_dim,
            "time_groups": str(self.time_groups),
            "time_groupby_factors": str(self.time_groupby_factors),
            "time_groupby_name": self.time_groupby_name,
        }
        ds_clim.to_netcdf(fpath)

    ##------------------------------------------------------------------------.
    def forecast(self, time, mean=True):
        """
        Forecast the climatology.

        Parameters
        ----------
        time : np.narray
            Timesteps at which retrieve the climatology.
        mean : bool, optional
            Wheter to forecast the climatological mean (when True) or variability.
            The default is True.

        Returns
        -------
        ds_forecast : xr.Dataset
            xarray Dataset with the forecasted climatology.

        """
        ##--------------------------------------------------------------------.
        # Check time_arr type
        if not isinstance(time, np.ndarray):
            raise TypeError("'time' must be a numpy array with np.datetime64 values.")
        if not np.issubdtype(time.dtype, np.datetime64):
            raise TypeError("The 'time' numpy array must have np.datetime64 values.")
        ##--------------------------------------------------------------------.
        # Define dims names
        groupby_dims = self.groupby_dims
        time_dim = self.time_dim
        dims = []
        dims.append(time_dim)
        if groupby_dims is not None:
            dims.append(*groupby_dims)
        ##--------------------------------------------------------------------.
        # Define dims shape
        dims_shape = []
        dims_shape.append(len(time))
        if groupby_dims is not None:
            for groupbydim in groupby_dims:
                dims_shape.append(self.mean_.dims[groupbydim])
        ##--------------------------------------------------------------------.
        # Define coords
        coords = []
        coords.append(time)
        if groupby_dims is not None:
            for groupbydim in groupby_dims:
                coords.append(self.mean_[groupbydim].values)
        ##--------------------------------------------------------------------.
        # Create DataArray of 1
        da_ones = xr.DataArray(data=np.ones(dims_shape), coords=coords, dims=dims)

        ##--------------------------------------------------------------------.
        # Create the climatological forecast
        time_groupby_info = get_time_groupby_idx(
            data=da_ones,
            time_dim=self.time_dim,
            time_groups=self.time_groups,
            time_groupby_factors=self.time_groupby_factors,
        )
        time_groupby_dims = time_groupby_info["time_groupby_dims"]
        time_groupby_idx = time_groupby_info["time_groupby_idx"]

        # - Mean
        ds_forecast = da_ones.groupby(time_groupby_idx) * self.mean_

        ##--------------------------------------------------------------------.
        # Remove time groups
        vars_to_remove = [self.time_groupby_name] + list(
            time_groupby_dims.data_vars.keys()
        )
        ds_forecast = ds_forecast.drop(vars_to_remove)

        ##--------------------------------------------------------------------.
        # Return the forecast
        return ds_forecast
