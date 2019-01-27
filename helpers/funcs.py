#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:16:20 2018

@author: berkkarahan
"""
import threading
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.stattools import adfuller
from dask import dataframe as dd

def resample(tseries, rate='15T', short_rate='S', max_gap=None):
    """ Resample (unevenly spaced) timeseries data linearly by first upsampling to a
        high frequency (short_rate) then downsampling to the desired rate.

    :param tseries: a pandas timeseries object
    :param rate: rate that tseries should be resampled to
    :param short_rate: intermediate upsampling rate; if None, smallest interval of tseries is used
    :param max_gap: null intervals larger than `max_gap` are being treated as missing
        data and not interpolated. if None, always interpolate. must be provided as pandas
        frequency string format, e.g. '6h'

    Copyright (c) 2017 WATTx GmbH
    License: Apache License
    """
    # return series if empty
    if tseries.empty:
        return tseries

    # check for datetime index
    assert isinstance(
        tseries.index[0], pd.tslib.Timestamp), 'Object must have a datetime-like index.'

    # sort tseries by time
    tseries.sort_index(inplace=True)

    # create timedelta from frequency string
    rate_delta = to_offset(rate).delta

    # compute time intervals
    diff = np.diff(tseries.index) / np.timedelta64(1, 's')

    if max_gap is not None:
        # identify intervals in tseries larger than max_gap
        idx = np.where(np.greater(diff, to_offset(max_gap).delta.total_seconds()))[0]
        start = tseries.index[idx].tolist()
        stop = tseries.index[idx + 1].tolist()
        # store start and stop indices of large intervals
        big_gaps = list(zip(start, stop))

    if short_rate is None:
        # use minimal nonzero interval of original series as short_rate
        short_rate = '%dS' % diff[np.nonzero(diff)].min()
        # create timedelta from frequency string
        short_rate_delta = to_offset(short_rate).delta
        # if smallest interval is still larger than rate, use rate instead
        if short_rate_delta > rate_delta:
            short_rate = rate
    else:
        # convert frequency string to timedelta
        short_rate_delta = to_offset(short_rate).delta
        # make sure entered short_rate is smaller than rate
        assert rate_delta >= short_rate_delta, 'short_rate must be <= rate'

    # upsample to short_rate
    tseries = tseries.resample(short_rate, how='mean').interpolate()

    # downsample to desired rate
    tseries = tseries.resample(rate, how='ffill')

    # replace values in large gap itervals with NaN
    if max_gap is not None:
        for start, stop in big_gaps:
            tseries[start:stop] = None

    return tseries

def resample_dask(tseries, rate='15T', short_rate='S', max_gap=None):
    """ Resample (unevenly spaced) timeseries data linearly by first upsampling to a
        high frequency (short_rate) then downsampling to the desired rate.

    :param tseries: a pandas timeseries object
    :param rate: rate that tseries should be resampled to
    :param short_rate: intermediate upsampling rate; if None, smallest interval of tseries is used
    :param max_gap: null intervals larger than `max_gap` are being treated as missing
        data and not interpolated. if None, always interpolate. must be provided as pandas
        frequency string format, e.g. '6h'

    Copyright (c) 2017 WATTx GmbH
    License: Apache License
    """
    # return series if empty
    if tseries.empty:
        return tseries

    # check for datetime index
    assert isinstance(
        tseries.index[0], pd.tslib.Timestamp), 'Object must have a datetime-like index.'

    # sort tseries by time
    tseries.sort_index(inplace=True)

    # create timedelta from frequency string
    rate_delta = to_offset(rate).delta

    # compute time intervals
    diff = np.diff(tseries.index) / np.timedelta64(1, 's')

    if max_gap is not None:
        # identify intervals in tseries larger than max_gap
        idx = np.where(np.greater(diff, to_offset(max_gap).delta.total_seconds()))[0]
        start = tseries.index[idx].tolist()
        stop = tseries.index[idx + 1].tolist()
        # store start and stop indices of large intervals
        big_gaps = list(zip(start, stop))

    if short_rate is None:
        # use minimal nonzero interval of original series as short_rate
        short_rate = '%dS' % diff[np.nonzero(diff)].min()
        # create timedelta from frequency string
        short_rate_delta = to_offset(short_rate).delta
        # if smallest interval is still larger than rate, use rate instead
        if short_rate_delta > rate_delta:
            short_rate = rate
    else:
        # convert frequency string to timedelta
        short_rate_delta = to_offset(short_rate).delta
        # make sure entered short_rate is smaller than rate
        assert rate_delta >= short_rate_delta, 'short_rate must be <= rate'

    tseries = dd.from_pandas(tseries, npartitions=32)
    tseries = tseries.resample(short_rate).mean().compute().interpolate()

    # downsample to desired rate
    tseries = tseries.resample(rate).ffill()

    # replace values in large gap itervals with NaN
    if max_gap is not None:
        for start, stop in big_gaps:
            tseries[start:stop] = None

    return tseries

def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """

    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test

def single_ts_split(df, test_size):
    """
        Perform train-test split with respect to time series structure
    """

    # get the index after which test set starts
    test_index = int(len(df)*(1-test_size))

    df_train = df.iloc[:test_index]
    df_test = df.iloc[test_index:]

    return df_train, df_test

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plotModelResults(model, X_train, X_test, y_train, y_test, tscv, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies

    """

    prediction = model.predict(X_test)

    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)

    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                                    cv=tscv,
                                    scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()

        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)

        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);

def plotCoefficients(model, X_train):
    """
        Plots sorted coefficient values of the model
    """

    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
    return coefs

#X and Y should be numpy arrays
def fit_model_cv(mdl, x, y, cv=5):
    def _fitmodel(mdl,x,y):
        return mdl.fit(x,y)

    kfold = TimeSeriesSplit(n_splits=cv)
    threadlist = []
    modlist = []
    train_ind = []
    holdout_ind = []
    predictions = []
    #score = []

    for tr_i, ho_i in kfold.split(x, y):
        train_ind.append(tr_i)
        holdout_ind.append(ho_i)
        cloned_mdl = clone(mdl)
        modlist.append(cloned_mdl)
        task = threading.Thread(target=_fitmodel, args=(cloned_mdl,x[tr_i],y[tr_i],))
        threadlist.append(task)

    for t in threadlist:
        t.start()

    for t in threadlist:
        t.join()

    #for i in range(0,5):
    #    m = modlist[i]
    #    predictions.append(m.predict(x[holdout_ind[i]]))
    #    #score.append(mean_squared_error(y[holdout_ind[i]], predictions[i]))
    #del threadlist, train_ind, holdout_ind, predictions
    #gc.collect()
    #print("Average mean_sq_error for models are: {}".format(np.mean(score)))
    return modlist


def iqr_filter_outliers(ts, multiplier=3, inplace=True):
    Q1 = ts.quantile(0.25)
    Q3 = ts.quantile(0.75)
    LB = Q1 - multiplier*(Q3-Q1)
    UB = Q3 + multiplier*(Q3-Q1)
    bl = (((ts < LB) | (ts > UB)))
    if inplace:
        ts[bl] == np.nan
        return ts
    else:
        return bl

def test_stationarity(timeseries, window=12):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(30, 20))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def readexcel_set_df_names(fname, sheet):
    df = pd.read_excel(fname, sheet_name=sheet)
    df.columns = df.iloc[0,:].values
    df.drop(df.index[0], inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df = df.astype(np.float)
    return df
