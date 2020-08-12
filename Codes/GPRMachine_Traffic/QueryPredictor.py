#!/usr/bin/python3
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : Mar/16/2020
# Author    : XHHAO
# Annotation: This file is for making prediction of target variable.
#===============================================================================
import numpy as np
from scipy.linalg import cholesky, cho_solve
#===============================================================================
#===============================================================================
def Predict(kernel, X_train, Y_train, X, sigma_n, cov_flag=False):
    K = kernel(X_train)
    K[np.diag_indices_from(K)] += sigma_n
    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), Y_train)
    K_trans = kernel(X, X_train)
    y_mean = K_trans.dot(alpha)
    if cov_flag == True:
        v = cho_solve((L, True), K_trans.T)
        y_cov = kernel(X) - K_trans.dot(v)
        #-----------------------------------------------------------------------
        return y_mean[0], y_cov[0,0]
    else:
        #-----------------------------------------------------------------------
            return y_mean[0]
#===============================================================================
#===============================================================================
def Prediction(TP, n_train, n_test, sigma_n, kernels):
    pred_Y = []
    n_map = n_test+1
    for p in range(0, n_test):
        pred_Y_temp = []
        for m in range(p+1, n_map):
            X_train = TP.variable_selection()
            Y_train = np.append(TP.Y_train, pred_Y[0:p])
            X = X_train[:n_train-m+p, ]
            Y = Y_train[m:]
            X_pred = X_train[n_train-m+p,]
            kernel = kernels[m]
            pred_y = Predict(kernel, X, Y, X_pred, sigma_n, False)
            pred_Y_temp.append(pred_y)
        y_mean = np.average(pred_Y_temp)
        pred_Y.append(y_mean)
    #---------------------------------------------------------------------------
    return np.asarray(pred_Y)
#===============================================================================
#===============================================================================