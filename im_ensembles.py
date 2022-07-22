#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time : 2021/7/15 10:41
@Author : Kunsong Zhao
"""

from imbalanced_ensemble.ensemble import *
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier


"""Under-sampling-based ensembles"""


def SPE(x_train, y_train, x_test):
    # clf = SelfPacedEnsembleClassifier(random_state=0)
    clf = SelfPacedEnsembleClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def BalCas(x_train, y_train, x_test):
    # clf = BalanceCascadeClassifier(random_state=0)
    clf = BalanceCascadeClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def BalRF(x_train, y_train, x_test):
    # clf = BalancedRandomForestClassifier(random_state=0)
    clf = BalancedRandomForestClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def EasyEn(x_train, y_train, x_test):
    # clf = EasyEnsembleClassifier(random_state=0)
    clf = EasyEnsembleClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def RUSB(x_train, y_train, x_test):
    # clf = RUSBoostClassifier(random_state=0)
    clf = RUSBoostClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def UnderBag(x_train, y_train, x_test):
    # clf = UnderBaggingClassifier(random_state=0)
    clf = UnderBaggingClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


"""Over-sampling-based ensembles"""


def OverBst(x_train, y_train, x_test):
    # clf = OverBoostClassifier(random_state=0)
    clf = OverBoostClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def SMOTEBst(x_train, y_train, x_test):
    # clf = SMOTEBoostClassifier(random_state=0)
    clf = SMOTEBoostClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


# def KmeansSMOTEBst(x_train, y_train, x_test):
#     clf = KmeansSMOTEBoostClassifier(random_state=0)
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     return y_pred


def OverBag(x_train, y_train, x_test):
    # clf = OverBaggingClassifier(random_state=0)
    clf = OverBaggingClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def SMOTEBag(x_train, y_train, x_test):
    # clf = SMOTEBaggingClassifier(random_state=0)
    clf = SMOTEBaggingClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


"""Reweighting-based ensembles"""


def AdaCost(x_train, y_train, x_test):
    # clf = AdaCostClassifier(random_state=0)
    clf = AdaCostClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def AdaUBst(x_train, y_train, x_test):
    # clf = AdaUBoostClassifier(random_state=0)
    clf = AdaUBoostClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def AsymBst(x_train, y_train, x_test):
    # clf = AsymBoostClassifier(random_state=0)
    clf = AsymBoostClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


"""Compatible ensembles"""


def CompAdaBst(x_train, y_train, x_test):
    # clf = CompatibleAdaBoostClassifier(random_state=0)
    clf = CompatibleAdaBoostClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def CompBag(x_train, y_train, x_test):
    # clf = CompatibleBaggingClassifier(random_state=0)
    clf = CompatibleBaggingClassifier(random_state=0, base_estimator=DecisionTreeClassifier())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


"""Others"""


def Bag(x_train, y_train, x_test):
    # bc = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree'), random_state=0)
    bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=0)

    bc.fit(x_train, y_train)
    y_pred = bc.predict(x_test)
    return y_pred


def BalBag(x_train, y_train, x_test):
    # bbc = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(
    #     n_neighbors=3, algorithm='ball_tree'),
    #                                 sampling_strategy='auto',
    #                                 replacement=False,
    #                                 random_state=0)

    bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=0)
    bbc.fit(x_train, y_train)
    y_pred = bbc.predict(x_test)
    return y_pred


def AdaBst(x_train, y_train, x_test):
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                             n_estimators=100, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred