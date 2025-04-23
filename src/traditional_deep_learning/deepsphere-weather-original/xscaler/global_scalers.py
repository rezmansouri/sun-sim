#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:40:29 2022

@author: ghiggi
"""
import time
import xarray as xr
import numpy as np
from .checks import (
    check_variable_dim,
    check_groupby_dims,
    check_rename_dict,
    _check_is_fitted,
    _check_save_fpath,
    get_xarray_variables,
)

# #####################
#### Global Scalers ###
# #####################
# - Statistics over all space, at each timestep: groupby_dims = "time"
# - Statistics over all timestep, at each pixel: groupby_dims = "node"


class GlobalStandardScaler:
    """StandardScaler aggregating over all dimensions (except variable_dim and groupby_dims)."""

    def __init__(
        self,
        data,
        variable_dim=None,
        groupby_dims=None,
        center=True,
        standardize=True,
        eps=0.0001,
        ds_scaler=None,
    ):
        # ds_scaler must not be specified. Use load_scaler(fpath) if you want to load an existing scaler from disk.
        ##--------------------------------------------------------------------.
        ### Load an already existing scaler (if ds_scaler is provided)
        if isinstance(ds_scaler, xr.Dataset):
            self.scaler_class = ds_scaler.attrs["scaler_class"]
            self.eps = ds_scaler.attrs["eps"]
            self.aggregating_dims = (
                ds_scaler.attrs["aggregating_dims"]
                if ds_scaler.attrs["aggregating_dims"] != "None"
                else None
            )
            self.groupby_dims = (
                ds_scaler.attrs["groupby_dims"]
                if ds_scaler.attrs["groupby_dims"] != "None"
                else None
            )
            self.center = True if ds_scaler.attrs["center"] == "True" else False
            self.standardize = (
                True if ds_scaler.attrs["standardize"] == "True" else False
            )
            self.fitted = True
            if self.center:
                self.mean_ = ds_scaler["mean_"].to_dataset(dim="variable")
            if self.standardize:
                self.std_ = ds_scaler["std_"].to_dataset(dim="variable")
        ##--------------------------------------------------------------------.
        ### Create the scaler object
        else:
            # Check center, standardize
            if not isinstance(center, bool):
                raise TypeError("'center' must be True or False'")
            if not isinstance(standardize, bool):
                raise TypeError("'standardize' must be True or False'")
            if not center and not standardize:
                raise ValueError(
                    "At least one between 'center' and 'standardize' must be 'true'."
                )
            ##--------------------------------------------------------------------.
            # Check data is an xarray Dataset or DataArray
            if not (isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray)):
                raise TypeError("'data' must be an xarray Dataset or xarray DataArray")
            ##--------------------------------------------------------------------.
            ## Checks for Dataset
            if isinstance(data, xr.Dataset):
                # Check variable_dim is not specified !
                if variable_dim is not None:
                    raise ValueError(
                        "'variable_dim' must not be specified for Dataset objects. Use groupby_dims instead."
                    )
            ##--------------------------------------------------------------------.
            # Checks for DataArray (and convert to Dataset)
            if isinstance(data, xr.DataArray):
                # Check variable_dim
                if variable_dim is None:
                    # If not specified, data name will become the dataset variable name
                    data = data.to_dataset()
                else:
                    variable_dim = check_variable_dim(
                        variable_dim=variable_dim, data=data
                    )
                    data = data.to_dataset(dim=variable_dim)
            ##---------------------------------------------------------------------.
            # Define groupby dimensions (over which to groupby)
            if groupby_dims is not None:
                groupby_dims = check_groupby_dims(
                    groupby_dims=groupby_dims, data=data
                )  # list (or None)
            self.groupby_dims = groupby_dims
            ##--------------------------------------------------------------------.
            # Retrieve dimensions over which to aggregate
            # - If DataArray, exclude variable_dims
            dims = np.array(list(data.dims))
            if groupby_dims is None:
                self.aggregating_dims = dims.tolist()
            else:
                self.aggregating_dims = dims[
                    np.isin(dims, groupby_dims, invert=True)
                ].tolist()
            ##--------------------------------------------------------------------.
            # Initialize
            self.scaler_class = "GlobalStandardScaler"
            self.data = data
            self.fitted = False
            self.eps = eps
            self.center = center
            self.standardize = standardize
            self.mean_ = None
            self.std_ = None
            # Save variable_dim if data is DataArray and using fit_transform()
            self.variable_dim = variable_dim

    ##------------------------------------------------------------------------.
    def fit(self, show_progress=True):
        """Fit the GlobalStandardScaler."""
        # ---------------------------------------------------------------------.
        # Checks
        if self.fitted:
            raise ValueError("The scaler has been already fitted!")
        # ---------------------------------------------------------------------.
        # Fit the scaler
        t_i = time.time()
        if self.center:
            self.mean_ = self.data.mean(self.aggregating_dims).compute()
        if self.standardize:
            self.std_ = self.data.std(self.aggregating_dims).compute()
        print("- Elapsed time: {:.2f}min".format((time.time() - t_i) / 60))
        # Set fitted flag to True
        self.fitted = True
        # del self.data

    def save(self, fpath, force=False):
        """Save the scaler object to disk in netCDF format."""
        ##--------------------------------------------------------------------.
        # Check
        _check_is_fitted(self)
        fpath = _check_save_fpath(fpath=fpath, force=force)
        ##--------------------------------------------------------------------.
        # Create xarray Dataset (to save as netCDF)
        mean_ = self.mean_.to_array()
        mean_.name = "mean_"
        std_ = self.std_.to_array()
        std_.name = "std_"
        # - Pack data into a Dataset based on center and standardize arguments
        if self.center and self.standardize:
            ds_scaler = xr.merge((mean_, std_))
        elif self.center:
            ds_scaler = mean_.to_dataset()
        else:
            ds_scaler = std_.to_dataset()
        ds_scaler.attrs = {
            "scaler_class": self.scaler_class,
            "eps": self.eps,
            "aggregating_dims": self.aggregating_dims
            if self.aggregating_dims is not None
            else "None",
            "groupby_dims": self.groupby_dims
            if self.groupby_dims is not None
            else "None",
            "center": str(self.center),
            "standardize": str(self.standardize),
        }
        ds_scaler.to_netcdf(fpath)

    ##------------------------------------------------------------------------.
    def transform(self, new_data, variable_dim=None, rename_dict=None):
        """Transform data using the fitted GlobalStandardScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted:
            raise ValueError("The GlobalStandardScaler need to be first fit() !.")
        ##--------------------------------------------------------------------.
        # Get variables to transform
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        if self.center:
            transform_vars = get_xarray_variables(self.mean_)
        else:
            transform_vars = get_xarray_variables(self.std_)
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data
        ##--------------------------------------------------------------------.
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim=variable_dim, data=new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)
            # Create dictionary for resetting dimensions name as original
            inv_rename_dict = {v: k for k, v in rename_dict.items()}
            # Rename dimensions
            new_data = new_data.rename(rename_dict)
        ##--------------------------------------------------------------------.
        # Check dimension name coincides
        new_data_dims = list(new_data.dims)
        if self.center:
            required_dims = list(self.mean_.dims)
        else:
            required_dims = list(self.std_.dims)
        if len(required_dims) >= 1:
            idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
            if np.any(idx_missing_dims):
                raise ValueError(
                    "Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(
                        np.array(required_dims)[idx_missing_dims]
                    )
                )
        ##--------------------------------------------------------------------.
        ## Transform variables
        new_data = new_data.copy()
        for var in transform_vars:
            if self.center:
                new_data[var] = new_data[var] - self.mean_[var]
            if self.standardize:
                new_data[var] = new_data[var] / (self.std_[var] + self.eps)
        ##--------------------------------------------------------------------.
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed:
            new_data = new_data.rename(inv_rename_dict)
        ##--------------------------------------------------------------------.
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return (
                    new_data.to_array(dim="variable", name=da_name)
                    .squeeze()
                    .drop("variable")
                    .transpose(*da_dims_order)
                )
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(
                    *da_dims_order
                )
        else:
            return new_data

    ##------------------------------------------------------------------------.
    def inverse_transform(self, new_data, variable_dim=None, rename_dict=None):
        """Inverse transform data using the fitted GlobalStandardScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted:
            raise ValueError("The GlobalStandardScaler need to be first fit() !.")
        ##--------------------------------------------------------------------.
        # Get variables to transform
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        if self.center:
            transform_vars = get_xarray_variables(self.mean_)
        else:
            transform_vars = get_xarray_variables(self.std_)
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data
        ##--------------------------------------------------------------------.
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim=variable_dim, data=new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)
            # Create dictionary for resetting dimensions name as original
            inv_rename_dict = {v: k for k, v in rename_dict.items()}
            # Rename dimensions
            new_data = new_data.rename(rename_dict)
        ##--------------------------------------------------------------------.
        # Check dimension name coincides
        new_data_dims = list(new_data.dims)
        if self.center:
            required_dims = list(self.mean_.dims)
        else:
            required_dims = list(self.std_.dims)
        if len(required_dims) >= 1:
            idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
            if np.any(idx_missing_dims):
                raise ValueError(
                    "Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(
                        np.array(required_dims)[idx_missing_dims]
                    )
                )
        ##--------------------------------------------------------------------.
        ## Transform variables
        new_data = new_data.copy()
        for var in transform_vars:
            if self.standardize:
                new_data[var] = new_data[var] * (self.std_[var] + self.eps)
            if self.center:
                new_data[var] = new_data[var] + self.mean_[var]
        ##--------------------------------------------------------------------.
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed:
            new_data = new_data.rename(inv_rename_dict)
        ##--------------------------------------------------------------------.
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return (
                    new_data.to_array(dim="variable", name=da_name)
                    .squeeze()
                    .drop("variable")
                    .transpose(*da_dims_order)
                )
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(
                    *da_dims_order
                )
        else:
            return new_data

    ##------------------------------------------------------------------------.
    def fit_transform(self):
        """Fit and transform directly the data."""
        ##--------------------------------------------------------------------.
        if self.fitted:
            raise ValueError(
                "The scaler has been already fitted. Please use .transform()."
            )
        ##--------------------------------------------------------------------.
        self.fit()
        return self.transform(new_data=self.data, variable_dim=self.variable_dim)


##----------------------------------------------------------------------------.
class GlobalMinMaxScaler:
    """MinMaxScaler aggregating over all dimensions (except variable_dim and groupby_dims)."""

    # TODO: feature_min, feature_max as dictionary per variable ...
    def __init__(
        self,
        data,
        variable_dim=None,
        groupby_dims=None,
        feature_min=0,
        feature_max=1,
        ds_scaler=None,
    ):
        # ds_scaler must not be specified. Use load_scaler(fpath) if you want to load an existing scaler from disk.
        ##--------------------------------------------------------------------.
        ### Load an already existing scaler (if ds_scaler is provided)
        if isinstance(ds_scaler, xr.Dataset):
            self.scaler_class = ds_scaler.attrs["scaler_class"]
            self.aggregating_dims = (
                ds_scaler.attrs["aggregating_dims"]
                if ds_scaler.attrs["aggregating_dims"] != "None"
                else None
            )
            self.groupby_dims = (
                ds_scaler.attrs["groupby_dims"]
                if ds_scaler.attrs["groupby_dims"] != "None"
                else None
            )
            self.feature_min = ds_scaler.attrs["feature_min"]
            self.feature_max = ds_scaler.attrs["feature_min"]
            self.fitted = True
            # Data
            self.min_ = ds_scaler["min_"].to_dataset(dim="variable")
            self.max_ = ds_scaler["max_"].to_dataset(dim="variable")
            self.range_ = ds_scaler["range_"].to_dataset(dim="variable")
            self.scaling = ds_scaler.attrs["scaling"]
        ##--------------------------------------------------------------------.
        ### Create the scaler object
        else:
            # Check feature_min, feature_max
            if not isinstance(feature_min, (int, float)):
                raise TypeError("'feature_min' must be a single number.'")
            if not isinstance(feature_max, (int, float)):
                raise TypeError("'feature_max' must be a single number.'")
            ##--------------------------------------------------------------------.
            # Check data is an xarray Dataset or DataArray
            if not (isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray)):
                raise TypeError("'data' must be an xarray Dataset or xarray DataArray")
            ##--------------------------------------------------------------------.
            ## Checks for Dataset
            if isinstance(data, xr.Dataset):
                # Check variable_dim is not specified !
                if variable_dim is not None:
                    raise ValueError(
                        "'variable_dim' must not be specified for Dataset objects. Use groupby_dims instead."
                    )
            ##--------------------------------------------------------------------.
            # Checks for DataArray (and convert to Dataset)
            if isinstance(data, xr.DataArray):
                # Check variable_dim
                if variable_dim is None:
                    # If not specified, data name will become the dataset variable name
                    data = data.to_dataset()
                else:
                    variable_dim = check_variable_dim(
                        variable_dim=variable_dim, data=data
                    )
                    data = data.to_dataset(dim=variable_dim)
            ##---------------------------------------------------------------------.
            # Define groupby dimensions (over which to groupby)
            if groupby_dims is not None:
                groupby_dims = check_groupby_dims(
                    groupby_dims=groupby_dims, data=data
                )  # list (or None)
            self.groupby_dims = groupby_dims
            ##--------------------------------------------------------------------.
            # Retrieve dimensions over which to aggregate
            # - If DataArray, exclude variable_dims
            dims = np.array(list(data.dims))
            if groupby_dims is None:
                self.aggregating_dims = dims.tolist()
            else:
                self.aggregating_dims = dims[
                    np.isin(dims, groupby_dims, invert=True)
                ].tolist()
            ##--------------------------------------------------------------------.
            # Initialize
            self.scaler_class = "GlobalMinMaxScaler"
            self.data = data
            self.fitted = False

            self.feature_min = feature_min
            self.feature_max = feature_max
            self.scaling = self.feature_max - self.feature_min

            # Save variable_dim if data is DataArray and using fit_transform()
            self.variable_dim = variable_dim

    ##------------------------------------------------------------------------.
    def fit(self, show_progress=True):
        """Fit the GlobalMinMaxScaler."""
        # ---------------------------------------------------------------------.
        # Checks
        if self.fitted:
            raise ValueError("The scaler has been already fitted!")
        # ---------------------------------------------------------------------.
        # Fit the scaler
        t_i = time.time()
        self.min_ = self.data.min(self.aggregating_dims).compute()
        self.max_ = self.data.max(self.aggregating_dims).compute()
        self.range_ = self.max_ - self.min_
        print("- Elapsed time: {:.2f}min".format((time.time() - t_i) / 60))
        self.fitted = True
        # del self.data

    def save(self, fpath, force=False):
        """Save the scaler object to disk in netCDF format."""
        # Check
        _check_is_fitted(self)
        fpath = _check_save_fpath(fpath=fpath, force=force)
        ##--------------------------------------------------------------------.
        # Create xarray Dataset (to save as netCDF)
        # - Convert to DataArray
        min_ = self.min_.to_array()
        min_.name = "min_"
        max_ = self.max_.to_array()
        max_.name = "max_"
        range_ = self.range_.to_array()
        range_.name = "range_"
        # - Pack data into a Dataset based on feature_min and feature_max arguments
        ds_scaler = xr.merge((min_, max_, range_))
        ds_scaler.attrs = {
            "scaler_class": self.scaler_class,
            "aggregating_dims": self.aggregating_dims
            if self.aggregating_dims is not None
            else "None",
            "groupby_dims": self.groupby_dims
            if self.groupby_dims is not None
            else "None",
            "scaling": self.scaling,
            "feature_min": self.feature_min,
            "feature_max": self.feature_max,
        }
        ds_scaler.to_netcdf(fpath)

    ##------------------------------------------------------------------------.
    def transform(self, new_data, variable_dim=None, rename_dict=None):
        """Transform data using the fitted GlobalStandardScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted:
            raise ValueError("The GlobalStandardScaler need to be first fit() !.")
        ##--------------------------------------------------------------------.
        # Get variables to transform
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        transform_vars = get_xarray_variables(self.min_)
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data
        ##--------------------------------------------------------------------.
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim=variable_dim, data=new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)
            # Create dictionary for resetting dimensions name as original
            inv_rename_dict = {v: k for k, v in rename_dict.items()}
            # Rename dimensions
            new_data = new_data.rename(rename_dict)
        ##--------------------------------------------------------------------.
        # Check dimension name coincides
        new_data_dims = list(new_data.dims)
        required_dims = list(self.min_.dims)
        if len(required_dims) >= 1:
            idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
            if np.any(idx_missing_dims):
                raise ValueError(
                    "Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(
                        np.array(required_dims)[idx_missing_dims]
                    )
                )
        ##--------------------------------------------------------------------.
        ## Transform variables
        new_data = new_data.copy()
        for var in transform_vars:
            new_data[var] = (new_data[var] - self.min_[var]) / self.range_[
                var
            ] * self.scaling + self.feature_min
        ##--------------------------------------------------------------------.
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed:
            new_data = new_data.rename(inv_rename_dict)
        ##--------------------------------------------------------------------.
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return (
                    new_data.to_array(dim="variable", name=da_name)
                    .squeeze()
                    .drop("variable")
                    .transpose(*da_dims_order)
                )
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(
                    *da_dims_order
                )
        else:
            return new_data

    ##------------------------------------------------------------------------.
    def inverse_transform(self, new_data, variable_dim=None, rename_dict=None):
        """Inverse transform data using the fitted GlobalMinMaxScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted:
            raise ValueError("The GlobalMinMaxScaler need to be first fit() !.")
        ##--------------------------------------------------------------------.
        # Get variables to transform
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        transform_vars = get_xarray_variables(self.min_)
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data
        ##--------------------------------------------------------------------.
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim=variable_dim, data=new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)
            # Create dictionary for resetting dimensions name as original
            inv_rename_dict = {v: k for k, v in rename_dict.items()}
            # Rename dimensions
            new_data = new_data.rename(rename_dict)
        ##--------------------------------------------------------------------.
        # Check dimension name coincides
        new_data_dims = list(new_data.dims)
        required_dims = list(self.min_.dims)
        if len(required_dims) >= 1:
            idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
            if np.any(idx_missing_dims):
                raise ValueError(
                    "Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(
                        np.array(required_dims)[idx_missing_dims]
                    )
                )
        ##--------------------------------------------------------------------.
        ## Transform variables
        new_data = new_data.copy()
        for var in transform_vars:
            new_data[var] = (new_data[var] - self.feature_min) * self.range_[
                var
            ] / self.scaling + self.min_[var]
        ##--------------------------------------------------------------------.
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed:
            new_data = new_data.rename(inv_rename_dict)
        ##--------------------------------------------------------------------.
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return (
                    new_data.to_array(dim="variable", name=da_name)
                    .squeeze()
                    .drop("variable")
                    .transpose(*da_dims_order)
                )
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(
                    *da_dims_order
                )
        else:
            return new_data

    ##------------------------------------------------------------------------.
    def fit_transform(self):
        """Fit and transform directly the data."""
        ##--------------------------------------------------------------------.
        if self.fitted:
            raise ValueError(
                "The scaler has been already fitted. Please use .transform()."
            )
        ##--------------------------------------------------------------------.
        self.fit()
        return self.transform(new_data=self.data, variable_dim=self.variable_dim)
