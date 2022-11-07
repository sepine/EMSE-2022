# !/usr/bin/env python
# encoding: utf-8

"""
@Description :
@Time : 2020/8/3 11:17
@Author : Kunsong Zhao
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

__author__ = 'kszhao'


class DataProcessor(object):

    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test
        self.scalar = StandardScaler(copy=True, with_mean=True, with_std=True)

    @staticmethod
    def split_feature_label(train):
        train = train.loc[:].values
        y_train = train[:, -1]
        x_train = train[:, : -1]
        return x_train, y_train

    def z_score_fit(self, x):
        x_scaled = self.scalar.fit_transform(x)
        return x_scaled

    def z_score_transform(self, x):
        x_scaled = self.scalar.transform(x)
        return x_scaled

    def process(self, is_validate=False):
        x_train, y_train = self.split_feature_label(self.train)
        x_test, y_test = self.split_feature_label(self.test)

        x_train_scaled = self.z_score_fit(x_train)
        x_test_scaled = self.z_score_transform(x_test)

        y_train = np.array(y_train, dtype=np.int)
        y_test = np.array(y_test, dtype=np.int)

        if not is_validate:
            return x_train, x_test, x_train_scaled, x_test_scaled, y_train, y_test

        x_valid, y_valid = self.split_feature_label(self.valid)
        x_valid_scaled = self.z_score_transform(x_valid)

        return x_train, x_valid, x_test, \
            x_train_scaled, x_valid_scaled, x_test_scaled, \
            y_train, y_valid, y_test
