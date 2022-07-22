# !/usr/bin/env python
# encoding: utf-8

"""
@Description :
@Time : 2020/8/3 1:05 下午
@Author : Kunsong Zhao
"""
from pandas import DataFrame, concat
from sklearn.metrics import auc, roc_auc_score
import numpy as np
import pandas as pd

__author__ = 'kszhao'


def get_metrics(y_true, y_pred):
    # 获取常见的4个值，用于系列指标计算
    TN, FP, FN, TP = np.fromiter((sum(
        bool(j >> 1) == bool(y_true[i]) and
        bool(j & 1) == bool(y_pred[i])
        for i in range(len(y_true))
    ) for j in range(4)), float)

    Accuracy = (TN + TP) / (TN + FP + FN + TP + 1e-8)
    Precision = TP / (TP + FP + 1e-8)
    # True Positive Rate
    Recall = TP / (TP + FN + 1e-8)
    # False Positive Rate
    FPR = FP / (FP + TN + 1e-8)

    # print("Precision", Precision)
    # print("Recall", Recall)

    g_mean = np.sqrt((TN / (TN + FP + 1e-8)) * (TP / (TP + FN + 1e-8)))
    Balance = 1 - np.sqrt((0 - FPR) ** 2 + (1 - Recall) ** 2) / np.sqrt(2)
    MCC = (TP * TN - FN * FP) / np.sqrt((TP + FN) * (TP + FP) * (FN + TN) * (FP + TN) + 1e-8)

    F_negative = harmonic_mean(TN / (TN + FN), TN / (TN + FP))

    # 当F_measure中θ值为2时
    F1 = 2 * Recall * Precision / (Recall + Precision + 1e-8)
    F2 = 5 * Recall * Precision / (4 * Recall + Precision + 1e-8)
    G_measure = 2 * Recall * (1 - FPR) / (Recall + (1 - FPR) + 1e-8)
    # NMI = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")

    # 返回所有指标值 vars() 函数返回对象object的属性和属性值的字典对象。
    y_pred = vars()
    # 该字典不返回'y_true', 'y_pred', "TN", "FP", "FN", "TP"这些key值
    return {k: y_pred[k] for k in reversed(list(y_pred)) if k not in ['y_true', 'y_pred', "TN", "FP", "FN", "TP", "FPR"]}


def evaluate(y_pred, y_test, xtest, columns=None, is_EA=False, threadshold=0.5):

    pred_class = [0 if i < threadshold else 1 for i in y_pred]
    # 将预测得到的结果类别pred_class与原始正确数据labels进行对比得到相应指标
    traditional_inditators = get_metrics(y_test, pred_class)
    # 根据预测结果，可计算ROC曲线下的区域
    auc = roc_auc_score(y_true=y_test, y_score=y_pred)

    traditional_inditators["auc"] = auc

    if not is_EA:
        return traditional_inditators

    else:
        if columns:
            test_df = get_loc_data(xtest, y_test, columns)
            test_df['pred'] = pred_class

            sorted_test_df = test_df.sort_values('loc')
            effort_indicators = effort_aware(sorted_test_df, positive_first(sorted_test_df))

            all_indicators = dict(traditional_inditators, **effort_indicators)

            return all_indicators


def norm_opt(*args) -> float:
    'Calculate the Alberg-diagram-based effort-aware indicator.'
    predict, optimal, worst = map(alberg_auc, args)
    return 1 - (optimal - predict) / (optimal - worst)


# df: target domain内被split分割的train data
def alberg_auc(df: DataFrame) -> float:
    'Calculate the area under curve in Alberg diagrams.'
    points = df[['loc', 'bug']].values.cumsum(axis=0)
    points = np.insert(points, 0, [0, 0], axis=0) / points[-1]
    return auc(*points.T)


# df: target domain内被split分割的test data
# EAPredict: 把上述test data正标签都放在前面，负标签放在后面
def effort_aware(df: DataFrame, EAPredict: DataFrame):
    """Calculate the effort-aware performance indicators."""
    EAOptimal = concat([df[df.bug == True], df[df.bug == False]])
    EAWorst = EAOptimal.iloc[::-1]

    M = len(df)
    N = sum(df.bug)
    m = threshold_index(EAPredict['loc'], 0.2)
    n = sum(EAPredict.bug.iloc[:m])
    for k, y in enumerate(EAPredict.bug):
        if y:
            break

    y = set(vars().keys())
    EA_Precision = n / m
    EA_Recall = n / N
    EA_F1 = harmonic_mean(EA_Precision, EA_Recall)
    EA_F2 = 5 * EA_Precision * EA_Recall / np.array(4 * EA_Precision + EA_Recall + 1e-8)
    PCI = m / M
    IFA = k
    P_opt = norm_opt(EAPredict, EAOptimal, EAWorst)
    M = vars()

    # print("EAPredict", EAPredict)
    # print("EAOptimal", EAOptimal)
    # print("EAWorst", EAWorst)
    return {k: M[k] for k in reversed(list(M)) if k not in y}


# Move the positive instances to the front of the dataset.
def positive_first(df: DataFrame) -> DataFrame:
    if sum(df.pred == df.bug) * 2 < len(df):
        df.pred = (df.pred == False)

    return concat([df[df.pred == True], df[df.pred == False]])


# 返回loc属性值内第一个比sum(loc)*percent大的下标值
def threshold_index(loc, percent: float) -> int:
    threshold = sum(loc) * percent
    for i, x in enumerate(loc):
        threshold -= x
        if threshold < 0:
            return i + 1


# 添加新的"loc"和"bug"属性
# TODO 修改loc的计算
def get_loc_data(target_datas, target_label, columns):
    target_label = target_label.astype('int')
    target_label = np.reshape(target_label, newshape=(len(target_label), 1))
    target_datas = np.hstack((target_datas, target_label))
    df = pd.DataFrame(target_datas, columns=columns)
    df["bug"] = df.pop(df.columns[-1])
    df["loc"] = df["ld"] + df["la"]
    # df["loc"] = df["ck_oo_numberOfLinesOfCode"]
    return df


def harmonic_mean(x, y, beta=1):
    beta *= beta
    return (beta + 1) * x * y / np.array(beta * x + y)