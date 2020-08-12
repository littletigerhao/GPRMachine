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
    def __init__(self, target):
        self.target = target
        self.name = target.split('_')[0]
        self.t_indx = int(target.split('_')[-1])-1
    #===========================================================================
    #===========================================================================
    def load_data(self, n_start, n_train, n_test, noise_level=0):
        self.noise_level = noise_level
        file_path = os.getcwd().split('/ProgramCodes')[0]+'/SourceFiles'
        data_file = file_path + '/' + self.name + '.mat'
        data = scio.loadmat(data_file)
        X = data['x']
        X += np.random.uniform(0, self.noise_level, np.shape(X))
        #=======================================================================
        self.X_train = X[n_start : n_start+n_train, ]
        self.Y_train = self.X_train[ : , self.t_indx]
        self.Y_test = X[n_start+n_train : n_start+n_train+n_test, self.t_indx]
    #===========================================================================
    #===========================================================================