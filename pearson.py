#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time : 2021/8/3 17:30
@Author : Kunsong Zhao
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def read_data(path):
    ret = pd.read_csv(path)
    return ret


def save_data(path, data, columns):
    data = pd.DataFrame(data, columns=columns)
    data.to_csv(path, sep='\t', index=False)


def cal_coefficient(x, y):
    co, p = pearsonr(x, y)
    return co


def processor(path, out_path):

    window_size = 20
    lag_size = 5

    data = read_data(path)
    stocks = sorted(set(data['stkcd'].iloc[:].values))

    all_cos = []

    for stock in stocks:

        sub_data = data[data['stkcd'] == stock]

        rrets = sub_data['rret'].iloc[:].values
        dates = sub_data['date'].iloc[:].values
        marrets = sub_data['marret'].iloc[:].values
        ts = sub_data['t'].iloc[:].values

        total_size = len(rrets)

        for idx in range(total_size - lag_size):
            if idx + window_size + lag_size < total_size:
                batch_x = rrets[idx: idx + window_size]
                batch_y = marrets[idx + lag_size: idx + window_size + lag_size]
                co = cal_coefficient(batch_x, batch_y)
                all_cos.append([stock, rrets[idx], dates[idx], marrets[idx], ts[idx], co])

            else:
                all_cos.append([stock, rrets[idx], dates[idx], ts[idx], marrets[idx],  ''])

        while len(all_cos) < total_size:
            idx = len(all_cos)
            all_cos.append([stock, rrets[idx], dates[idx], ts[idx], marrets[idx],  ''])

    columns = ['stkcd', 'rret', 'date', 'marret', 't', 'coefficient']
    save_data(out_path, all_cos, columns)


if __name__ == '__main__':
    processor(path='./did3.csv',
              out_path='./co.txt')
