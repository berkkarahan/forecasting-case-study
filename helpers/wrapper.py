#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:25:27 2018

@author: berkkarahan
"""
import numpy as np
from luminol.anomaly_detector import AnomalyDetector
from random import seed, gauss

def rand_gauss(mu, sigma, rand_seed, out_shape):
    randlist = []
    seed(rand_seed)
    for _ in range(out_shape+1):
        randlist.append(gauss(mu, sigma))

class LuminolWrapper:
    def __init__(self, pandas_ts):

        self.pandas_ts = pandas_ts
        self._idx_to_date = {}
        self._ts_dict = {}

        self._anoms_lmn = None
        self._an_idx_built = False
        self._anoms = []

        self._adclass = AnomalyDetector
        self._ad = None

        self._build_luminol_tsdict()
        self._initAnomalyDetector()
        self._get_luminol_anomalies()
        self._buildAnomalyIndexes()



    def _build_luminol_tsdict(self):
        for idx, s_tup in enumerate(self.pandas_ts.iteritems()):
            self._idx_to_date[idx] = s_tup[0]
            self._ts_dict[idx] = s_tup[1]

    def _initAnomalyDetector(self):
            self._ad = self._adclass(self._ts_dict)

    def _get_luminol_anomalies(self):
        self._anoms_lmn = self._ad.get_anomalies()

    def _buildAnomalyIndexes(self):
        for a in self._anoms_lmn:
            if a.start_timestamp == a.end_timestamp:
                self._anoms.append(a.start_timestamp)
            else:
                for j in range(a.start_timestamp, a.end_timestamp + 1):
                    self._anoms.append(j)
        self._an_idx_built = True

    @staticmethod
    def rand_gauss(mu, sigma, rand_seed, out_shape):
        randlist = []
        seed(rand_seed)
        for _ in range(out_shape):
            randlist.append(gauss(mu, sigma))
        return randlist

    def getDateTime(self):
        dt = []
        if self._an_idx_built:
            for aidx in self._anoms:
                dt.append(self._idx_to_date[aidx])
        return dt

    def booleanMaskDateTime(self):
        dt = self.getDateTime()
        return np.array([True if x in dt else False for x in self.pandas_ts.index])

    def imputeAnomalies(self, randseed, method='gauss'):
        bm = self.booleanMaskDateTime()
        _bmi = ~np.copy(bm)
        _shp = sum(bm==True)
        _mean = self.pandas_ts[_bmi].mean()
        if method == 'gauss':
            _std = self.pandas_ts[_bmi].std()
            _imps = self.rand_gauss(_mean, _std, randseed, _shp)
        elif method == 'mean':
            _imps = np.zeros((sum(bm == True)))
            _imps = _mean
        ts_cp = self.pandas_ts.copy()
        ts_cp[bm] = _imps
        return ts_cp
