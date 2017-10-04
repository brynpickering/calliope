"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

sampling.py
~~~~~~~

Methods for sampling from distributions to produce scenario-specific datasets.
"""
import os

from . import exceptions
from . import time_clustering
from scipy.cluster import hierarchy

import pandas as pd
import xarray as xr
import numpy as np
import scipy.cluster.vq as vq

def get_clusters(dataframe, func, **kwargs):
    """
    Create clusters using the same methods as found in time_clustering, but
    with differently formatted input dataframe.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        index as dates, columns as timesteps
    func: string
        'kmeans', 'hierarchical', or 'degree_day', following the available
        clustering methods in time_clustering.
    **kwargs: Additional inputs for each clustering method

    Returns
    -------
    clustereds : pandas.Series, index as timesteps (same index as input
        dataframe), values are cluster groups
    """
    def _kmeans(data, k):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Should be normalized
        k : int
            Number of cluster groups
        Returns
        -------
        clusters : pandas.DataFrame
            Indexed by timesteps and with locations as columns, giving cluster
            membership for first timestep of each day.
        centroids

        """
        X = data.dropna().values
        centroids, distortion = vq.kmeans(X, k)

        # Determine the cluster membership of each day
        day_clusters = vq.vq(X, centroids)[0]

        return pd.Series(day_clusters, index=data.index)

    def _degree_days(data, method='heating', ref_T=15.5, clusters=[0, 5, 10],
                     T_filepath='/temperatures.csv'):
        """
        Get clusters from weather data, picking days based on the
        heating/cooling degree day

        Parameters
        ----------
        data : pandas.DataFrame
            Should be normalized
        method : string; default = 'heating'
            Either 'heating' (Tdd - T if T < Tdd else 0)
            or 'cooling' (T - Tdd if T > Tdd else 0)
        ref_T : float; default = 15.5
            Reference temperature, in degC, corresponding to Tdd in heating degree
            day calculations
        clusters : list; default = [0, 5, 10]
            Each list element refers to lower bound of degree day clusters.
        T_filepath : string; Default = '/temperatures.csv'
            filepath to 'temperatures.csv', usually held in the model data folder

        Returns
        -------
        clusters : pandas.DataFrame
            Indexed by timesteps and with locations as columns, giving cluster
            membership for first timestep of each day.
        """
        big_M = [1e3] # to catch any values above the last cluster lower bound

        temperatures = pd.read_csv(T_filepath, header=None, index_col=0)
        temperatures.index = pd.to_datetime(temperatures.index)
        temperatures = temperatures.resample('1H').mean() # 24 timesteps per day
        temperatures_mean = temperatures.groupby(temperatures.index.dayofyear).mean()
        if method == 'heating':
            degree_hour = (ref_T - temperatures) / 24
        elif method == 'cooling':
            degree_hour = (temperatures - ref_T) / 24
        degree_hour[degree_hour < 0] = 0
        degree_day = degree_hour.resample('1D').sum().iloc[:, 0]

        degree_day_mean = degree_day.groupby(degree_day.index.dayofyear).mean()
        for nan_day in degree_day[degree_day.isnull()].index:
            degree_day[nan_day] = degree_day_mean[nan_day.dayofyear]
        labels = [i for i in range(len(clusters))]
        if data.index.dtype == np.int64: # mean data
            degree_day_clusters = pd.cut(degree_day_mean, bins=clusters+big_M,
                                         include_lowest=True, labels=labels)
        elif isinstance(data.index, pd.DatetimeIndex): # mean data
            degree_day_clusters = pd.cut(degree_day, bins=clusters+big_M,
                                         include_lowest=True, labels=labels)
        return degree_day_clusters

    def _hierarchical(data, k, max_d=None):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Should be normalized
        max_d : float or int, optional. Default = None
            Max distance for returning clusters. Overrides k, if defined.
        k : int, optional
            Number of desired clusters. Overidden by max_d, if it is defined

        Returns
        -------
        clusters : pandas.DataFrame
            Indexed by timesteps and with locations as columns, giving cluster
            membership for first timestep of each day.
        """
        X = data.dropna().values
        # Generate the linkage matrix
        Z = hierarchy.linkage(X, 'ward')
        if max_d:
            # Get clusters based on maximum distance
            clusters = hierarchy.fcluster(Z, max_d, criterion='distance')
        elif k:
            # Get clusters based on number of desired clusters
            clusters = hierarchy.fcluster(Z, k, criterion='maxclust')
        return pd.Series(clusters, index=data.index)

    # Normalize the data, to be between 0 and 1
    ds = dataframe.copy(deep=True)
    max_vals = np.fabs(ds).max()
    normalised_data = ds / max_vals

    # cluster by desired function:
    clusters = locals()['_' + func](normalised_data, **kwargs)
    clusters = pd.DataFrame(clusters)
    return clusters

def get_multivariate_normal_samples(Mean, covariance, scenarios, matrix='full'):
    """
    Randomly sample a multivariate normal curve (diagonal or full) to get a
    timeseries curve for each scenario.

    Parameters
    ----------
    Mean: pandas DataFrame; index = timesteps, columns = technology/location
          sets (depending on data being sampled)
        Values of DataFrame are the Mean for that timestep (e.g. hour) over the
        entire available dataset for a given technology/location
    covariance: xarray DataArray; dimensions = (tech/loc, timestep, timestep)
        For each technology/location, the covariance relating each timestep to
        each other (e.g. each hour in the day). (timestep, timestep) should be
        a square matrix
    scenarios: int;
        Number of scenarios to sample
    matrix: str or int, optional; default = 'full'
         either 'full' or the number of adjacencies in a diagonal covariance matrix

    Returns
    -------
    X: xarray DataArray; dimensions = (tech/loc, timesteps, scenarios)

    """
    x = []
    for i in Mean.columns:
        MU = Mean[i]
        SIGMA = covariance.loc[i, :, :].to_pandas()
        if isinstance(matrix, np.int):  # diagonal matrix
            R = np.diag(np.ones(len(MU.index)))

            matrix += 1

            for adjacency in range(matrix):
                adjacency += 1
                step = 1 - (adjacency / matrix)
                R_adjacent_pos = np.diag(np.ones(len(MU.index)-adjacency)*step,
                                         adjacency)
                R_adjacent_neg = np.diag(np.ones(len(MU.index)-adjacency)*step,
                                         -adjacency)
                R = R + R_adjacent_pos + R_adjacent_neg

            SIGMA = R * SIGMA
        samples = np.random.multivariate_normal(MU, SIGMA, scenarios).T
        if any(MU < 0) and (samples > 0).any():
            exceptions.ModelWarning('positive values sampled for negative '
                                    'resource. Setting to zero')
            samples[samples > 0] = 0
        elif any(MU > 0) and (samples < 0).any():
            exceptions.ModelWarning('negative values sampled for positive '
                                    'resource. Setting to zero')
            samples[samples < 0] = 0
        x.append(samples)  # creating a 3D numpy array

    X = xr.DataArray(x, coords=[(covariance.dims[0], Mean.columns),  # name of first column depends on datatype (e.g. locations = x, techs = y)
                                ('t', Mean.index),
                                ('s', [i+1 for i in range(scenarios)])])
    return X

def get_normal_samples(Mean, variance, scenarios, skew=0.0, coord='unknown'):
    """
    Randomly sample a normal curve (diagonal or full) to get a
    timeseries curve for each scenario.
    Parameters:
    -----------
    Mean: float or pandas DataFrame;
        Mean value of normal distribution, as a single value or as a DataFrame,
        giving a mean value for each dataset of interest, e.g. technologies,
        locations (columns), at each timestep of interest (index).
    variance: float or pandas DataFrame;
        Variance of normal distribution, as a single value or as a DataFrame,
        giving the variance for each dataset of interest, e.g. technologies,
        locations (columns), at each timestep of interest (index).
    scenarios: int;
        Number of scenarios to sample
    skew: float or pandas DataFrame; Default = 0
        Degree of tail skew in the normal distribution, as a single value or as a DataFrame,
        giving the variance for each dataset of interest, e.g. technologies,
        locations (columns), at each timestep of interest (index).
        Positive = skewed in positive x-axis, Negative = skewed in favour of negative x-axis
    coord: str; Default = 'unknown'
        name of column set for mean, variance, and skew (if DataFrame)
    Returns:
    -------
    If Mean/variance is float:
        X: array, length = scenarios.
    if Mean/variance is DataFrame:
        X: xarray DataArray; dimensions = (tech/loc, timesteps, scenarios)

    """
    def random_skew(Mean, variance, scenarios, skew=0.0):
        # adapted from https://stackoverflow.com/questions/36200913/generate-n-random-numbers-from-a-skew-normal-distribution-using-numpy
        SIGMA = np.divide(skew, np.sqrt(np.add(np.power(skew, 2), 1)))
        if isinstance(Mean, np.float):
            u0 = np.random.randn(scenarios)
            v = np.random.randn(scenarios)
        else:
            u0 = np.random.randn(scenarios, len(Mean))
            v = np.random.randn(scenarios, len(Mean))
        u1 = np.multiply(
            np.add(
                np.multiply(SIGMA, u0),
            np.multiply(
                np.sqrt(
                    np.add(1.0, -np.power(SIGMA, 2))
                    ), v
                )
            ), np.sqrt(variance))
        u1[u0 < 0] *= -1
        u1 = np.add(u1, Mean)
        return u1

    if type(Mean) != type(variance) or type(Mean) != type(skew):
        return exceptions.ModelError('for normal distribution random sampling, '
        'mean, skew and variance must all be float or all be pandas DataFrames')
    if isinstance(Mean, np.float):
        X = random_skew(Mean, variance, scenarios, skew)
    elif isinstance(Mean, pd.DataFrame):
        x = []
        for i in Mean.columns:
            x.append(random_skew(Mean[i].values, variance[i].values, scenarios,
                                 skew[i].values).T)
        X = xr.DataArray(x, coords=[(coord, Mean.columns),  # name of first column depends on datatype (e.g. locations = x, techs = y)
                                ('t', Mean.index),
                                ('s', [i+1 for i in range(scenarios)])])
    return X

def create_scenarios(model, data, method, scenarios=1):
    """
    direct link to core.py, taking in xarray data and a dict of method information
    and returning an updated DataArray, now including the scenario dimension.
    """
    scenario_array = xr.DataArray(np.ones(scenarios),
                                  dims=('scenarios'),
                                  coords=[('scenarios',
                                          [i+1 for i in range(scenarios)])])
    scenario_data = data.copy(deep=True)
    if method.distribution == 'multivariate_normal':

        if 'raw_data' in method.keys():
            x_map = method.raw_data.get('x_map')  # dict of mapping model locations to filenames
            locations = pd.Series(x_map) if x_map else data.x.to_pandas()
            # Get a DataArray that is indexed by dates and locations,
            # actual values don't matter as they'll be overwritten
            dates = data.resample('1D', 't').t.to_pandas().index

            for x, x_map in locations.items():
                raw_data = pd.read_csv(os.path.join(model.config_model.data_path,
                                                    method.raw_data.folder,
                                                    x_map + '.csv'),
                                                    header=0,
                                                    index_col=0,
                                                    parse_dates=True).dropna()
                raw_data_clusters = get_clusters(raw_data, method.cluster.func,
                    **method.cluster.get('arguments'))
                # Raw data is probably over several years, so get the mean of
                # each day in the year
                raw_data_mean = raw_data.groupby(raw_data.index.dayofyear).mean()

                mean_clusters = get_clusters(raw_data_mean, method.cluster.func,
                    **method.cluster.get('arguments')).loc[dates.dayofyear]  # extract relevant days
                # We have to remove the possibility of getting positive/negative
                # results from negative/positive data, by taking the natural log

                for cluster in np.unique(raw_data_clusters):
                    daily_index = dates[mean_clusters.iloc[:, 0] == cluster]
                    if len(daily_index) == 0:
                        continue
                    hourly_index = time_clustering._hourly_from_daily_index(daily_index)

                    Mean = pd.DataFrame(
                        raw_data[raw_data_clusters.iloc[:, 0] == cluster].mean(axis=0),
                        columns=[x_map])
                    covariance = xr.DataArray(
                        [raw_data[raw_data_clusters.iloc[:, 0]
                                  == cluster].cov().values],
                        dims=['x', 'timesteps', 'timesteps'],
                        coords=[('x', [x_map]), ('timesteps', raw_data.columns),
                                ('timesteps', raw_data.columns)])
                    samples = get_multivariate_normal_samples(
                        Mean, covariance, scenarios,
                        matrix=method.get('matrix', 'full'))
                    for dim in samples.dims:
                        if len(samples[dim]) == 1 and dim != 's':
                            samples = samples.squeeze([dim])
                    samples_datetime = xr.broadcast(samples,
                        data.t.loc[dict(t=daily_index)]
                        .rename({'t':'dates'}))[0].stack(datetime=('dates', 't'))

                    scenario_data.loc[dict(t=hourly_index, x=x)] = \
                        samples_datetime.transpose('datetime', 's').values

            return scenario_data

        elif 'attributes' in method.keys():
            attributes = xr.open_dataarray(os.path.join(
                model.config_model.data_path, method.attributes)) # mean, covariance and time clusters. method.attributes is a filename
            datetimes = time_clustering._hourly_from_daily_index(
                attributes.clusters.dates) # index is dates only, here we expand to hourly timestamps
            clusters_datetime = (attributes.clusters.reindex(datetimes)
                                 .fillna(method='ffill').astype(int))
            for cluster in attributes.cluster:
                scenario_data.loc[dict(t=
                    clusters_datetime[clusters_datetime==cluster].index)] = (
                        get_multivariate_normal_samples(
                        attributes['mean'].loc[dict(cluster=cluster)].to_pandas(),
                        attributes['covariance'].loc[dict(cluster=cluster)],
                        scenarios, matrix=method.matrix))
            return scenario_data
        else:
            return exceptions.ModelError('Raw data or processed distribution '
                                         'attributes must be defined for '
                                         'scenario generation of {}'.format(data.name))
