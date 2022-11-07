# !/usr/bin/env python
# encoding: utf-8

"""
@Description : Load Dataset as DataFrame
@Time : 2020/8/3 10:25
@Author : Kunsong Zhao
"""

import os
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


__author__ = 'kszhao'



class DataLoader(object):

    def __init__(self, params):
        self.base_path = params['base_path']
        self.ext = params['ext']   # '.xlsx'
        self.flag = params['flag']   # 'contains_bug'
        self.split_ratio = params['split_ratio']
        # self.is_validate = params['is_validate']
        self.all_paths, self.dataset_names = self.__get_dataset_names(self.base_path)

    def __get_dataset_names(self, base_path):
        all_paths = []
        dataset_names = []
        for r, d, f in os.walk(base_path):
            for file in f:
                if self.ext in file:
                    all_paths.append(os.path.join(r, file))
                    dataset_names.append(file[: -len(self.ext)])
        return all_paths, dataset_names

    def read_xlsx(self, path):
        df = pd.read_excel(path)
        # Eliminate NAN
        df.fillna(value=0, inplace=True)
        return df

    def load_data(self, name):
        if self.ext == '.xlsx':
            data = self.read_xlsx(self.base_path + name + self.ext)
        columns = data.columns.tolist()
        return data, columns

    def get_dataset_names(self):
        return self.dataset_names, self.all_paths

    def split_dataset(self, df: DataFrame, values=[1, 0]) -> tuple:
        positives, negatives = (df[df[self.flag] == v] for v in values)
        (p_train, p_test), (n_train, n_test) = map(
            lambda dataset: train_test_split(dataset, test_size=self.split_ratio, shuffle=True, random_state=None),
            (positives, negatives))

        return p_train.append(n_train), p_test.append(n_test)

    def build_dataset(self, data, is_validate=False):
        train, test = self.split_dataset(data)
        if not is_validate:
            return train, None, test

        train, valid = self.split_dataset(train)
        return train, valid, test
