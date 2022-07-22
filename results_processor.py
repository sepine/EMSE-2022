#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time : 2021/7/17 11:23
@Author : Kunsong Zhao
"""
import os
import json
import pandas as pd


ext = '.xlsx'

indicator_columns = ["Precision", "Recall", "F_negative", "F1", "F2",
                     "g_mean", "Balance", "MCC", "G_measure", "auc"]

dataset_names = ['codec', 'collections', 'io', 'jsoup', 'jsqlparser', 'mango', 'ormlite']


def read_xlsx(path):
    df = pd.read_excel(path)
    # Eliminate NAN
    df.fillna(value=0, inplace=True)
    return df


def load_data(path):
    data = read_xlsx(path)
    columns = data.columns.tolist()
    return data, columns


def get_dataset_names(base_path):
    all_paths = []
    mean_paths = []
    all_dataset_names = []
    mean_dataset_names = []
    for r, d, f in os.walk(base_path):
        for file in f:
            if ext in file:
                if file.startswith('all'):
                    all_paths.append(os.path.join(r, file))
                    all_dataset_names.append(file[4: -len(ext)])
                elif file.startswith('mean'):
                    mean_paths.append(os.path.join(r, file))
                    mean_dataset_names.append(file[5: -len(ext)])
    return all_paths, all_dataset_names, mean_paths, mean_dataset_names


def process(path, out_path):
    all_paths, all_names, mean_paths, mean_names = get_dataset_names(path)

    print(mean_names)
    print(len(mean_names))
    print(mean_paths)
    print(len(mean_paths))
    print(all_names)
    print(len(all_names))
    print(all_paths)
    print(len(all_paths))

    # process mean values, split by each indicator
    mean_results = {indicator: {dataset: {method: []
                                          for method in mean_names}
                                for dataset in dataset_names}
                    for indicator in indicator_columns}

    for i in range(len(mean_paths)):
        data, columns = load_data(mean_paths[i])
        datasets = data['dataset'].tolist()
        for col in columns:
            if col == 'dataset':
                continue
            indicator_values = data[col].tolist()

            for idx in range(len(datasets)):
                mean_results[col][datasets[idx]][mean_names[i]] = indicator_values[idx]

    with open(out_path + 'mean.json', 'w', encoding='utf-8') as fw:
        json.dump(mean_results, fw)

    # process all values, split by each indicator
    all_results = {indicator: {dataset: {method: {i: [] for i in range(50)}
                                          for method in all_names}
                                for dataset in dataset_names}
                    for indicator in indicator_columns}

    for i in range(len(all_paths)):
        data, columns = load_data(all_paths[i])
        datasets = data['dataset'].tolist()
        for col in columns:
            if col == 'dataset':
                continue
            indicator_values = data[col].tolist()

            for idx in range(len(datasets)):
                values_list = indicator_values[idx].replace('[', '').replace(']', '').split(',')
                for idx_val in range(len(values_list)):
                    all_results[col][datasets[idx]][all_names[i]][idx_val] = values_list[idx_val].strip()

    with open(out_path + 'all.json', 'w', encoding='utf-8') as fw:
        json.dump(all_results, fw)


if __name__ == '__main__':
    # process(path='./results_v2', out_path='./processed_v2/')
    process(path='./results_cross', out_path='./processed_cross/')