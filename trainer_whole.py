#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time : 2021/4/4 15:35
@Author : Kunsong Zhao
"""

import os
import pandas as pd
import collections
import numpy as np
from data_processor import DataProcessor
from evaluation import evaluate
from classifiers import *
from im_ensembles import *
from im_weights import CSDT
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, params):
        self.params = params

        self.steps = params['steps']
        self.out_path = params['out_path']

        self.indicator_columns = ["Precision", "Recall", "F_negative", "F1", "F2",
                                  "g_mean", "Balance", "MCC", "G_measure", "auc"]

        # self.data_loader = DataLoader(self.params)
        # self.dataset_names, self.all_paths = self.data_loader.get_dataset_names()

        # self.classifiers = [DT, RF, LR, SVM, NB, MLP, NN, Ripper]
        self.classifiers = [DT]

        # self.sampling_methods = [ClusCent, RUS, InsHard, NearMs,
        #                          TomekLs, EditNN, RepEditNN, Aknn,
        #                          OneSS, CondensNN, NNCRule, Adasyn,
        #                          ROS, Smote, BorSmote, SvmSmote,
        #                          SmoteENN, SmoteTomek, NONE]

        # self.ensemble_methods = [SPE, BalCas, BalRF, EasyEn, RUSB, UnderBag,
        #                          OverBst, SMOTEBst, OverBag, SMOTEBag, AdaCost,
        #                          AdaUBst, AsymBst, CompAdaBst,
        #                          CompBag, Bag, BalBag, AdaBst]

        self.ensemble_methods = [OverBag, UnderBag, AsymBst]

        # self.cs_methods = [CSNN, CSLR, CSDT, CSRF, CSSVM]
        self.cs_methods = [CSDT]

        self.mean_results = collections.defaultdict(list)
        self.all_results = collections.defaultdict(list)

    def test(self, ratio):

        # load data
        df = pd.read_excel('./whole/whole.xlsx')
        positives, negatives = (df[df['flag'] == v] for v in [1, 0])
        # print(positives.shape)
        # print(negatives.shape)
        # print(positives)

        # data, columns = self.data_loader.load_data(name)

        all_algo_indicators = collections.defaultdict(dict)

        for i in range(self.steps):

            # shuffle
            positives = positives.sample(frac=1.0).reset_index(drop=True)
            negatives = negatives.sample(frac=1.0).reset_index(drop=True)
            # print(positives)

            # split
            train_pos = positives[:200]
            neg_count = int(200 * (ratio + 1))
            train_neg = negatives[:neg_count]
            train = pd.concat([train_pos, train_neg])
            test_pos = positives[200:]
            test_neg = negatives[neg_count:]
            test = pd.concat([test_pos, test_neg])

            train = train.sample(frac=1.0).reset_index(drop=True)
            test = test.sample(frac=1.0).reset_index(drop=True)

            print(train.shape)
            print(test.shape)

            print('**********************************************', str(i + 1))
            print('******************************************************')
            print('******************************************************')

            # for each epoch, call the build method for random splitting.
            # train, _, test = self.data_loader.build_dataset(data)

            data_processor = DataProcessor(train, None, test)
            x_train, x_test, x_train_scaled, x_test_scaled, y_train, y_test = data_processor.process()

            # (1) sampling based methods.
            print('*********** Sampling based methods *******************')
            print('******************************************************')
            print('******************************************************')

            for clf in self.classifiers:  # only DT
                method_name = clf.__name__
                y_pred = clf(x_train_scaled, y_train, x_test_scaled)

                all_indicators = evaluate(y_pred, y_test, x_test, columns=None, is_EA=False)
                for key in all_indicators:
                    if key in all_algo_indicators[method_name].keys():
                        all_algo_indicators[method_name][key].append(all_indicators[key])
                    else:
                        all_algo_indicators[method_name][key] = [all_indicators[key]]

            # (2) ensemble based methods
            print('*********** Ensemble based methods *******************')
            print('******************************************************')
            print('******************************************************')
            for ensemble in self.ensemble_methods:  # only for OBag and UBag

                print('******************************', ensemble.__name__)

                y_pred = ensemble(x_train_scaled, y_train, x_test_scaled)

                all_indicators = evaluate(y_pred, y_test, x_test, columns=None, is_EA=False)
                for key in all_indicators:
                    if key in all_algo_indicators[ensemble.__name__].keys():
                        all_algo_indicators[ensemble.__name__][key].append(all_indicators[key])
                    else:
                        all_algo_indicators[ensemble.__name__][key] = [all_indicators[key]]

            # (3) Cost sensitive based methods
            print('*********** Cost sensitive based methods *******************')
            print('******************************************************')
            print('******************************************************')
            for cs in self.cs_methods:  # CSDT
                print('******************************', cs.__name__)

                cls_wights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
                y_pred = cs(x_train_scaled, y_train, x_test_scaled, cls_wights)

                all_indicators = evaluate(y_pred, y_test, x_test, columns=None, is_EA=False)
                for key in all_indicators:
                    if key in all_algo_indicators[cs.__name__].keys():
                        all_algo_indicators[cs.__name__][key].append(all_indicators[key])
                    else:
                        all_algo_indicators[cs.__name__][key] = [all_indicators[key]]

        return all_algo_indicators

    def test_all(self):

        for ratio in range(10):

            print("********************Ratio***************", str(ratio + 1))

            total_indicators = self.test(ratio)
            for algo_name in total_indicators.keys():
                indicator_data = total_indicators[algo_name]

                dict_columns = list(indicator_data.keys())

                new_columns = [dict_columns.index(col) for col in self.indicator_columns]
                indicator_values = np.array(list(indicator_data.values()))[new_columns, :]

                all_row = [ratio] + indicator_values.tolist()
                self.all_results[algo_name].append(all_row)

                # Calculate mean value
                mean_value = indicator_values.transpose()
                mean_value = mean_value.mean(axis=0)

                mean_row = [ratio] + mean_value.tolist()
                self.mean_results[algo_name].append(mean_row)

        # save the average results
        for algo_name in list(self.mean_results.keys()):
            print("Now is print", algo_name, "mean model")
            one_data = self.mean_results[algo_name]

            one_data = pd.DataFrame(one_data, columns=["dataset"] + self.indicator_columns)
            final_path = os.path.join(self.out_path, "mean_" + algo_name + ".xlsx")
            one_data.to_excel(final_path, index=False)

        # save all the results
        for algo_name in list(self.all_results.keys()):
            print("Now is print", algo_name, "all model")
            one_data = self.all_results[algo_name]
            one_data = pd.DataFrame(one_data, columns=["dataset"] + self.indicator_columns)
            final_path = os.path.join(self.out_path, "all_" + algo_name + ".xlsx")
            one_data.to_excel(final_path, index=False)
