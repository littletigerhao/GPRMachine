#!/usr/bin/python3
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : May/07/2020
# Author    : XHHAO
# Annotation: This file is for training the mappings with GPR(basic/consistent).
#===============================================================================
import numpy as np
from GPRModeller import GeneralGPR
#===============================================================================
#===============================================================================
class TrainingProcess():
    #===========================================================================
    #===========================================================================
    def __init__(self, X_train, Y_train, n_test, dropout, sigma_n, n_run, target):
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_train = np.shape(X_train)[0]
        self.n_test = n_test
        self.n_map = n_test + 1
        self.dropout = dropout
        self.sigma_n = sigma_n
        self.n_run = n_run
        self.X_dim = X_train.shape[1]
        self.target = target
    #===========================================================================
    #===========================================================================
    def get_kernels(self, kernel_flag):
        if kernel_flag == 'BasicTrain':
            return self.kernels_BT
        elif kernel_flag == 'ConsisTrain':
            return self.kernels_CT
        else:
            print('Wrong flag was input, please manually check.')
    #===========================================================================
    #===========================================================================
    def variable_selection(self):
        #-----------------------------------------------------------------------
        if self.target == 'Lorentz_16' or self.target == 'LorentzPart_16':
            indx_keep = [9,10,11,13,15,16,17,21,22,27,36,37,38,40,47,73,75,80,85]
            #indx_keep = [i for i in range(90)]
        elif self.target == 'Lorentz_17':
            indx_keep = [10,16,21,28,36,37,39,40,45,46,49,51,66,76,82,85,87,88]
        elif self.target == 'Lorentz_18':
            indx_keep = [10,11,16,17,22,27,36,37,38,39,40,41,47,53,73,74,86,89]
        #-----------------------------------------------------------------------
        elif self.target == 'Wind_P_58':
            indx_keep = [56,57,58,84,95,96]
        #-----------------------------------------------------------------------
        elif self.target == 'Typhoon_1':
            indx_keep = [0]
        elif self.target == 'Typhoon_2':
            indx_keep = [i for i in range(2024)]
            indx_keep = [1,16,18,20,46]
        #-----------------------------------------------------------------------
        elif self.target == 'Traffic_21':
            indx_keep = [i for i in range(207)]
        #-----------------------------------------------------------------------
        elif self.target == 'Stock_2':
            indx_keep = [i for i in range(1130)]
            indx_keep = [0,1]
        #-----------------------------------------------------------------------
        elif 'HUAWEI_R' or 'HUAWEI_I' in self.target:
            indx_keep = [i for i in range(192)]
        #-----------------------------------------------------------------------
        elif 'GeneExpression' in self.target:
            indx_keep = [i for i in range(4900,5000)]
        #-----------------------------------------------------------------------
        else:
            indx_keep = []
            for kernel in self.kernels_PT:
                lsv = kernel.get_params()['k1__k2__length_scale']
                indx_keep_temp = [i for i in range(self.X_dim) if lsv[i] < 100]
                indx_keep = np.append(indx_keep, indx_keep_temp)
            indx_keep = list(set(list(indx_keep)))
        #-----------------------------------------------------------------------
        indx_all = [i for i in range(self.X_dim)]
        indx_del = [item for item in indx_all if item not in indx_keep]
        X_train = np.delete(self.X_train, indx_del, axis=1)
        #-----------------------------------------------------------------------
        return X_train
    #===========================================================================
    #===========================================================================
    def pre_training(self):
        print('>> Pre-training is in processing ...')
        kernels = []
        for m in range(0, self.n_map):
            X_train = self.X_train[:self.n_train-m, ]
            Y_train = self.Y_train[m:,]
            GPR = GeneralGPR(X_train, Y_train, self.dropout, self.sigma_n, self.n_run)
            GPR.fit()
            kernel = GPR.get_kernel()
            kernels.append(kernel)
        #-----------------------------------------------------------------------
        self.kernels_PT = kernels
    #===========================================================================
    #===========================================================================
    def basic_training(self):
        print('>> Basic training is in processing ...')
        kernels = []
        for m in range(0, self.n_map):
            print('>> Now training for %d-th mapping ...'%(m))
            Xs_train = self.variable_selection()
            X_train = Xs_train[:self.n_train-m, ]
            Y_train = self.Y_train[m:,]
            GPR = GeneralGPR(X_train, Y_train, self.dropout, self.sigma_n, self.n_run)
            GPR.fit()
            kernel = GPR.get_kernel()
            kernels.append(kernel)
        #-----------------------------------------------------------------------
        self.kernels_BT = kernels
    #===========================================================================
    #===========================================================================
    def consistent_training(self):
        print('>> Consistent training is in processing ...')
        #=======================================================================
        kernels = [i for i in range(self.n_map)]
        pred_y_mean_list = []
        #-----------------------------------------------------------------------
        for ps in range(0, self.n_test):
            pred_Y = []
            for m in range(ps+1, self.n_map):
                Xs_train = self.variable_selection()
                X_train = Xs_train[:self.n_train-m+ps, ]
                Y_train = np.append(self.Y_train[m:,], pred_y_mean_list[:ps])
                X = Xs_train[self.n_train-m+ps,]
                GPR = GeneralGPR(X_train, Y_train, self.dropout, self.sigma_n, self.n_run)
                GPR.fit()
                kernel = GPR.get_kernel()
                kernels[m] = kernel
                pred_y = GPR.Predict(X)
                pred_Y.append(pred_y)
            pred_y_mean = np.average(pred_Y)
            pred_y_mean_list.append(pred_y_mean)
        #-----------------------------------------------------------------------
        self.kernels_CT = kernels
#===============================================================================
#===============================================================================