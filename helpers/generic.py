#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:55:17 2018

@author: berkkarahan
"""
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
from sklearn.base import clone

class GenericFitter:
    def __init__(self, ts, model, nshift=4):
        self.ts = ts
        self.model = clone(model)
        self.nshift = nshift
        
    def _shift(self,ts):
        ts_ = ts.shift(-self.nshift)
        return ts_.dropna()
    
    def fit(self):
        X = self.ts.copy().values.reshape(-1,1)
        X = X[:-self.nshift]
        y = self.ts.shift(-self.nshift).dropna().values.reshape(-1,1)
        self.model.fit(X,y)
        
    def predict(self, n_periods):
        modulo = (n_periods) / self.nshift
        niter = np.int(np.ceil(modulo))
        preds = list()
        
        x = self.ts[-self.nshift:].values
        
        for i in range(niter):
            x = x.reshape(-1,1)
            y_pred = self.model.predict(x)
            preds.append(y_pred)
            x = y_pred
        
        preds = np.concatenate(preds).ravel()
        return preds[:n_periods]
    
    def multifit_predict(self, n_periods):
        
        
        modulo = (n_periods) / self.nshift
        niter = np.int(np.ceil(modulo))
        preds = list()
        
        X_fit = self.ts.values.reshape(-1,1)
        X_fit = X_fit[:-self.nshift]
        y_fit = self.ts.shift(-self.nshift).dropna().values.reshape(-1,1)
        
        x = self.ts[-self.nshift:].values
        
        for i in range(niter):
            
            mdl = clone(self.model)
            mdl.fit(X_fit, y_fit)     
            x = x.reshape(-1,1)
            y_pred = self.model.predict(x)
            preds.append(y_pred)
            X_fit = np.vstack((X_fit,y_pred))
            y_fit = shift(X_fit, self.nshift)[:-self.nshift]
            X_fit = X_fit[:-self.nshift]
            x = X_fit[-self.nshift:]
            
        preds = np.concatenate(preds).ravel()
        return preds[:n_periods]
        
        
        
        