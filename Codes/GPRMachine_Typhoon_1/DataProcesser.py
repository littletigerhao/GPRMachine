#!/usr/bin/python3
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : Apr/29/2020
# Author    : XHHAO
# Annotation: This file is for processing the data.
#===============================================================================
import os
import numpy as np
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
        if self.name == 'Typhoon':
            data = np.loadtxt(data_file)
            X = data
        #=======================================================================
        self.X_dim = np.shape(X)[1]
        self.X_train = X[n_start : n_start+n_train, ]
        self.Y_train = self.X_train[ : , self.t_indx]
        self.Y_test = X[n_start+n_train : n_start+n_train+n_test, self.t_indx]
    #===========================================================================
    #===========================================================================