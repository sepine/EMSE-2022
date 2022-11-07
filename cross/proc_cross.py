# !/usr/bin/env python
# encoding: utf-8

"""
@Description :
@Time : 6/29/22 8:26 PM
@Author : Kunsong Zhao
"""

import os
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


PROJECTS = ['codec', 'collections', 'io', 'jsoup', 'jsqlparser', 'mango', 'ormlite']


def read_xlsx(path):
    df = pd.read_excel(path)
    # Eliminate NAN
    df.fillna(value=0, inplace=True)
    return df


def merge2cross():
    # df_1 = read_xlsx(path=os.path.join('../datasets/codec.xlsx'))
    df_2 = read_xlsx(path=os.path.join('../datasets/collections.xlsx'))
    df_3 = read_xlsx(path=os.path.join('../datasets/io.xlsx'))
    df_4 = read_xlsx(path=os.path.join('../datasets/jsoup.xlsx'))
    df_5 = read_xlsx(path=os.path.join('../datasets/jsqlparser.xlsx'))
    df_6 = read_xlsx(path=os.path.join('../datasets/mango.xlsx'))
    df_7 = read_xlsx(path=os.path.join('../datasets/ormlite.xlsx'))

    columns = df_2.columns.tolist()
    df_all = pd.concat([df_2, df_3, df_4, df_5, df_6, df_7])

    print(df_all)
    print(df_all.shape)
    print(columns)

    df_all.to_excel('./train_codec.xlsx', index=False)


if __name__ == '__main__':
    merge2cross()