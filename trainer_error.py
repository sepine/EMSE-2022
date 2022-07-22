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
from data_loader import DataLoader
from data_processor import DataProcessor
from evaluation import evaluate
from classifiers import *
from im_samplings import *
from im_ensembles import *
from im_weights import *
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, params):
        self.params = params

        self.steps = params['steps']
        self.out_path = params['out_path']

        self.indicator_columns = ["Precision", "Recall", "F_negative", "F1", "F2",
                                  "g_mean", "Balance", "MCC", "G_measure", "auc"]

        self.data_loader = DataLoader(self.params)
        self.dataset_names, self.all_paths = self.data_loader.get_dataset_names()

        # self.classifiers = [DT, RF, LR, SVM, NB, MLP, NN, Ripper]
        self.classifiers = [DT]

        # self.sampling_methods = [ClusCent, RUS, InsHard, NearMs,
        #                          TomekLs, EditNN, RepEditNN, Aknn,
        #                          OneSS, CondensNN, NNCRule, Adasyn,
        #                          ROS, Smote, BorSmote, SvmSmote,
        #                          SmoteENN, SmoteTomek, NONE]

        self.ensemble_methods = [UnderBag, OverBag, AsymBst]

        # self.cs_methods = [CSNN, CSLR, CSDT, CSRF, CSSVM]
        self.cs_methods = [CSDT]

        self.mean_results = collections.defaultdict(list)
        self.all_results = collections.defaultdict(list)

    def test(self, name):

        data, columns = self.data_loader.load_data(name)

        all_algo_indicators = collections.defaultdict(dict)

        for i in range(self.steps):

            print('**********************************************', str(i + 1))
            print('******************************************************')
            print('******************************************************')

            # for each epoch, call the build method for random splitting.
            train, _, test = self.data_loader.build_dataset(data)

            data_processor = DataProcessor(train, None, test)
            x_train, x_test, x_train_scaled, x_test_scaled, y_train, y_test = data_processor.process()

            # (1) sampling based methods.
            for clf in self.classifiers:
                method_name = clf.__name__
                print('============' + method_name + '============')
                y_pred = clf(x_train_scaled, y_train, x_test_scaled)

                cm = confusion_matrix(y_test, y_pred)
                tp = cm[1][1]
                fp = cm[0][1]
                fn = cm[1][0]
                tn = cm[0][0]

                print('TP:' + str(tp) + '  FP:' + str(fp) + '  TN:' + str(tn) + '  fn:' + str(fn))

            # (2) ensemble based methods
            for ensemble in self.ensemble_methods:

                print('===========' + ensemble.__name__ + '=============')

                y_pred = ensemble(x_train_scaled, y_train, x_test_scaled)
                cm = confusion_matrix(y_test, y_pred)
                tp = cm[1][1]
                fp = cm[0][1]
                fn = cm[1][0]
                tn = cm[0][0]

                print('TP:' + str(tp) + '  FP:' + str(fp) + '  TN:' + str(tn) + '  fn:' + str(fn))

            # (3) Cost sensitive based methods
            for cs in self.cs_methods:
                print('===========' + cs.__name__ + '=============')

                cls_wights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
                y_pred = cs(x_train_scaled, y_train, x_test_scaled, cls_wights)

                cm = confusion_matrix(y_test, y_pred)
                tp = cm[1][1]
                fp = cm[0][1]
                fn = cm[1][0]
                tn = cm[0][0]

                print('TP:' + str(tp) + '  FP:' + str(fp) + '  TN:' + str(tn) + '  fn:' + str(fn))

    def test_all(self):

        for name in self.dataset_names:
            print('==============' + name + '===============')
            self.test(name)
