"""
@Description :
@Time : 2020/8/3 11:19
@Author : Kunsong Zhao
"""

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import wittgenstein as lw



def NB(X, y, xtest):
    gnb = GaussianNB()
    gnb.fit(X, y)
    y_pred = gnb.predict(xtest)
    return y_pred


def DT(X, y, xtest):
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(xtest)
    return y_pred


def SVM(X, y, xtest):
    clf = SVC()
    clf.fit(X, y)
    y_pred = clf.predict(xtest)
    return y_pred


def LR(X, y, xtest):
    l_r = LogisticRegression(random_state=0)
    l_r.fit(X, y)
    y_pred = l_r.predict(xtest)
    return y_pred


def RF(X, y, xtest):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    y_pred = clf.predict(xtest)
    return y_pred


def NN(X, y, xtest):
    neigh = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
    neigh.fit(X, y)
    y_pred = neigh.predict(xtest)
    return y_pred


def MLP(X, y, xtest):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=0)
    clf.fit(X, y)
    y_pred = clf.predict(xtest)
    return y_pred


def Ripper(X, y, xtest):
    clf = lw.RIPPER()
    clf.fit(X, y)
    y_pred = clf.predict(xtest)
    return y_pred
