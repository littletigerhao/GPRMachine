#!/usr/bin/python  
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : Mar/12/2020
# Author    : XHHAO
# Annotation: This file is gaussian process regression, training part.
#===============================================================================
import os
import shutil
import sklearn
from DataProcesser import DataProcesser
from ParamTrainer import TrainingProcess
from QueryPredictor import Prediction
from ResultEvaluator import PerformanceEvaluation
from ResultSaver import SaveResults
#===============================================================================
#===============================================================================
if __name__ == '__main__':
    #===========================================================================
    # ---------------------- sk-learn Version Information ---------------------#
    #===========================================================================
    print('==='*25)
    print('Training with the use of sklearn, version:', sklearn.__version__)
    #===========================================================================
    # ------------------------- Define parameters -----------------------------#
    #===========================================================================
    n_start = 2
    n_train = 48
    n_test = 22
    n_run = 100
    sigma_n = 1e-10
    sigma_n = 0
    dropout = 0.2
    #=======================================================================
    # ------------------------ Initialization -----------------------------#
    #=======================================================================
    target = 'Typhoon_2'
    DP = DataProcesser(target)
    DP.load_data(n_start, n_train, n_test)
    current_path = os.getcwd().split('/ProgramCodes')[0]
    result_dir = current_path + '/ResultFiles/GPRMachine/'+target
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    #=======================================================================
    # -----------------------Train and Prediction -------------------------#
    #=======================================================================
    print('==='*25)
    print('Training for typhoon eye latitude ...')
    #-------------------------------------------------------------------
    X_train = DP.X_train
    Y_train = DP.Y_train
    Y_test = DP.Y_test
    #-------------------------------------------------------------------
    TP = TrainingProcess(X_train, Y_train, n_test, dropout, sigma_n, n_run, target)
    #TP.consistent_training()
    #kernels = TP.get_kernels('ConsisTrain')
    TP.basic_training()
    kernels = TP.get_kernels('BasicTrain')
    Y_pred = Prediction(TP, n_train-6, n_test, sigma_n, kernels)
    #-------------------------------------------------------------------
    print('>> Evaluation ...')
    performance = PerformanceEvaluation(Y_test, Y_pred)
    #-------------------------------------------------------------------
    print('>> Prediction done! Write to files ...')
    SaveResults(Y_train, Y_test, Y_pred, performance, 0, result_dir, target)
    print('==='*25)
#===============================================================================
#===============================================================================