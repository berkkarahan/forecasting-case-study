#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:39:47 2018

@author: berkkarahan
"""

import numpy as np
import pandas as pd
import collections
from itertools import count

class SimpleAnomalies:
    
    def __init__(self, Y, window_size, sigma):
        self.y = Y
        self.window_size = window_size
        self.sigma = sigma
        self.explained_anoms = None
        self.anom_idx = None

    def _moving_average(self):
        """ Computes moving average using discrete linear convolution of two one dimensional sequences.
        Args:
        -----
                data (pandas.Series): independent variable
                window_size (int): rolling window size

        Returns:
        --------
                ndarray of linear convolution

        References:
        ------------
        [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
        [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html

        """

        _window = np.ones(int(self.window_size))/float(self.window_size)
        return np.convolve(self.y, _window, 'same')

    def buildIndexes(self):
        if self.explain_anomalies is not None:
            orddct = self.explain_anomalies['anomalies_dict']
            self.anom_idx = [x for x in orddct]

    def imputeAnomaliesRolling(self, inplace=False):
        if not inplace:
            y_c = self.y.copy()
            rmean = y_c.rolling(self.window_size, center=True, min_periods=1).mean()
            y_c.update(rmean)
            return y_c
        if inplace:
            rmean = self.y.rolling(self.window_size, center=True, min_periods=1).mean()
            self.y.update(rmean)
            return self.y


    def explain_anomalies(self):
        """ Helps in exploring the anamolies using stationary standard deviation
        Args:
        -----
            y (pandas.Series): independent variable
            window_size (int): rolling window size
            sigma (int): value for standard deviation

        Returns:
        --------
            a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
            containing information about the points indentified as anomalies

        """
        avg = self._moving_average().tolist()
        residual = self.y - avg
        # Calculate the variation in the distribution of the residual
        std = np.std(residual)
        self.explain_anomalies = {'standard_deviation': round(std, 3),
                'anomalies_dict': collections.OrderedDict([(index, y_i) for
                                                           index, y_i, avg_i in zip(count(), self.y, avg)
                  if (y_i > avg_i + (self.sigma*std)) | (y_i < avg_i - (self.sigma*std))])}
        return self

    def explain_anomalies_rolling_std(self):
        """ Helps in exploring the anamolies using rolling standard deviation
        Args:
        -----
            y (pandas.Series): independent variable
            window_size (int): rolling window size
            sigma (int): value for standard deviation

        Returns:
        --------
            a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
            containing information about the points indentified as anomalies
        """
        avg = self._moving_average()
        avg_list = avg.tolist()
        residual = self.y - avg
        # Calculate the variation in the distribution of the residual
        testing_std = residual.rolling(self.window_size).std()
        #pd.rolling_std(residual, window_size)
        testing_std_as_df = pd.DataFrame(testing_std)
        rolling_std = testing_std_as_df.replace(np.nan,
                                      testing_std_as_df.ix[self.window_size - 1]).round(3).iloc[:,0].tolist()
        std = np.std(residual)
        self.explain_anomalies = {'stationary standard_deviation': round(std, 3),
                'anomalies_dict': collections.OrderedDict([(index, y_i)
                                                           for index, y_i, avg_i, rs_i in zip(count(),
                                                                                               self.y, avg_list, rolling_std)
                  if (y_i > avg_i + (self.sigma * rs_i)) | (y_i < avg_i - (self.sigma * rs_i))])}
        return self
