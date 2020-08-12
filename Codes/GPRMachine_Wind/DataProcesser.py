#!/usr/bin/python3
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : Apr/29/2020
# Author    : XHHAO
# Annotation: This file is for processing the data.
#===============================================================================
import os
import numpy as np
import scipy.io as scio
#===============================================================================
#===============================================================================
class DataProcesser():
    #===========================================================================
    #===========================================================================
    def __init__(self, target, ):
        self.name = target.split('_')[0]
        self.t_indx = int(target.split('_')[1])-1
    #===========================================================================
    #===========================================================================
    def load_data(self, n_start, n_train, n_test, noise_level=0):
        self.noise_level = noise_level
        file_path = os.getcwd().split('/ProgramCodes')[0]+'/SourceFiles'
        data_file = file_path + '/' + self.name + '.mat'
        #=======================================================================
        if self.name == 'Lorentz':
            data = scio.loadmat(data_file)
            X = data['x']
            X += np.random.uniform(0, self.noise_level, np.shape(X))
        #=======================================================================
        elif self.name == 'Wind':
            data = np.loadtxt(data_file)
            X = data
            '''
            n_col = np.shape(X)[1]
            X1 = np.append(X[1:,], np.zeros((1, n_col)), axis=0)
            X2 = np.append(X[2:,], np.zeros((2, n_col)), axis=0)
            X3 = np.append(X[3:,], np.zeros((3, n_col)), axis=0)
            X4 = np.append(X[4:,], np.zeros((4, n_col)), axis=0)
            X5 = np.append(X[5:,], np.zeros((5, n_col)), axis=0)
            X = (X+X1+X2+X3+X4+X5)/6
            '''
        #=======================================================================
        elif self.name == 'Typhoon':
            data = np.loadtxt(data_file)
            X = data
            n_col = np.shape(X)[1]
            X1 = np.append(X[1:,], np.zeros((1, n_col)), axis=0)
            X2 = np.append(X[2:,], np.zeros((2, n_col)), axis=0)
            X3 = np.append(X[3:,], np.zeros((3, n_col)), axis=0)
            X4 = np.append(X[4:,], np.zeros((4, n_col)), axis=0)
            X5 = np.append(X[5:,], np.zeros((5, n_col)), axis=0)
            X = (X+X1+X2+X3+X4+X5)/6
        #=======================================================================
        # t_indx {Lorentz:15,16,17 | Wind: 57 | Typhoon: 0,1}
        self.X_dim = np.shape(X)[1]
        self.X_train = X[n_start : n_start+n_train, ]
        self.Y_train = self.X_train[ : , self.t_indx]
        self.Y_test = X[n_start+n_train : n_start+n_train+n_test, self.t_indx]
    #===========================================================================
    #===========================================================================