# !/usr/bin/env python
# encoding: utf-8

"""
@Description :
@Time : 6/7/22 3:40 PM
@Author : Kunsong Zhao
"""


# from keras.models import Sequential
# from keras.layers import Dense, Activation
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# def CSNN(x_train, y_train, x_test, weights):
#     model = Sequential()
#     model.add(Dense(40, input_shape=(89,)))
#     model.add(Activation('relu'))
#     model.add(Dense(1, activation='sigmoid'))
#
#     model.compile(optimizer='rmsprop', loss='binary_crossentropy')
#
#     history = model.fit(x_train, y_train, epochs=20, batch_size=64, class_weight=weights)
#     y_pred = model.predict_classes(x_test).flatten()
#
#     return y_pred


def CSDT(x_train, y_train, x_test, weights):
    clf = DecisionTreeClassifier(class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def CSSVM(x_train, y_train, x_test, weights):
    clf = SVC(class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def CSLR(x_train, y_train, x_test, weights):
    l_r = LogisticRegression(random_state=0, class_weight='balanced')
    l_r.fit(x_train, y_train)
    y_pred = l_r.predict(x_test)
    return y_pred


def CSRF(x_train, y_train, x_test, weights):
    clf = RandomForestClassifier(max_depth=2, random_state=0, class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred