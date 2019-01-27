#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 21:24:24 2018

@author: berkkarahan
"""

import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

class ARIMASearch:
    
    def __init__(self, X, max_params=3, cv=5):
        
        self.X = X
        self.cv = cv
        self._r = range(0,max_params)
        self.params = list(itertools.product(self._r,self._r,self._r))
        self.tscv = TimeSeriesSplit(n_splits=cv)
        self.resdf = pd.DataFrame()
        
    def fitoneparam(self,param, tscv=True):
        
        cvaic = []
        cvmse = []
        mlist = []
        
        for tr_i, ho_i in self.tscv.split(self.X):
            try:
                mod = sm.tsa.ARIMA(self.X[tr_i],
                                   order=param)
                results = mod.fit()
                cvaic.append(results.aic)
                y_pred = mod.predict(param, start=min(self.X[ho_i].index), end=max(self.X[ho_i].index))
                mse = mean_squared_error(self.X[ho_i], y_pred)
                cvmse.append(mse)
            except:
                continue

        return (np.mean(cvaic), np.mean(cvmse), )
    
    def fitall(self):
        for param in self.params:
            res = self.fitoneparam(param)
            self.resdf['p'] = param[0]
            self.resdf['d'] = param[1]
            self.resdf['q'] = param[2]
            self.resdf['avg_cv_aic'] = res[0]
            self.resdf['avg_cv_mse'] = res[1]