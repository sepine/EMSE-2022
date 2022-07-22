#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time : 2021/7/15 11:03
@Author : Kunsong Zhao
"""

from imbalanced_ensemble.sampler.under_sampling import *
from imbalanced_ensemble.sampler.over_sampling import *
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, ADASYN


"""Under-sampling Samplers"""


def ClusCent(x_train, y_train):
    cc = ClusterCentroids(random_state=42)
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def RUS(x_train, y_train):
    cc = RandomUnderSampler(random_state=42)
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def InsHard(x_train, y_train):
    cc = InstanceHardnessThreshold(random_state=42)
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def NearMs(x_train, y_train):
    cc = NearMiss()
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def TomekLs(x_train, y_train):
    cc = TomekLinks()
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def EditNN(x_train, y_train):
    cc = EditedNearestNeighbours()
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def RepEditNN(x_train, y_train):
    cc = RepeatedEditedNearestNeighbours()
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def Aknn(x_train, y_train):
    cc = AllKNN()
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def OneSS(x_train, y_train):
    cc = OneSidedSelection(random_state=42)
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def CondensNN(x_train, y_train):
    cc = CondensedNearestNeighbour(random_state=42)
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def NNCRule(x_train, y_train):
    cc = NeighbourhoodCleaningRule()
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


# def BalCasUS(x_train, y_train):
#     cc = BalanceCascadeUnderSampler(random_state=42)
#     x_res, y_res = cc.fit_resample(x_train, y_train, sample_weight=None)
#     return x_res, y_res


# def SPUS(x_train, y_train):
#     cc = SelfPacedUnderSampler(random_state=42)
#     x_res, y_res = cc.fit_resample(x_train, y_train)
#     return x_res, y_res


"""Over-sampling Samplers"""


def Adasyn(x_train, y_train):
    cc = ADASYN(random_state=42)
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def ROS(x_train, y_train):
    cc = RandomOverSampler(random_state=42)
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


# def KmsSmote(x_train, y_train):
#     cc = KMeansSMOTE(random_state=42)
#     x_res, y_res = cc.fit_resample(x_train, y_train)
#     return x_res, y_res


def Smote(x_train, y_train):
    cc = SMOTE(random_state=42)
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def BorSmote(x_train, y_train):
    cc = BorderlineSMOTE(random_state=42)
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def SvmSmote(x_train, y_train):
    cc = SVMSMOTE(random_state=42)
    x_res, y_res = cc.fit_resample(x_train, y_train)
    return x_res, y_res


def SmoteENN(X, y):
    smote_enn = SMOTEENN(random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled


def SmoteTomek(X, y):
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    return X_resampled, y_resampled


def NONE(X, y):
    return X, y
