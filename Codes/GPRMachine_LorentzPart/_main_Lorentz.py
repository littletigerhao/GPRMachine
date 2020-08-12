#!/usr/bin/python3
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : Mar/12/2020
# Author    : XHHAO
# Annotation: This file is the entrance of Lorentz.
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
    print('Program running with the use of sklearn, version:', sklearn.__version__)
    #=======================================================================
    # ----------------------- Define parameters ---------------------------#
    #=======================================================================
    target = 'Lorentz_16'
    n_start = 0
    n_train = 30
    n_test = 25
    n_run = 1            # this parameter can be increased up to 200
    dropout = 0.2
    noise_level = 0
    #=======================================================================
    # ------------------------ Initialization -----------------------------#
    #=======================================================================
    DP = DataProcesser(target)
    DP.load_data(n_start, n_train, n_test, noise_level)
    current_path = os.getcwd().split('/ProgramCodes')[0]
    result_dir = current_path + '/ResultFiles/GPRMachine/'+target+'Part'
    result_dir += '/NoiseLevel_' + str(noise_level)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    #=======================================================================
    # -----------------------Train and Prediction -------------------------#
    #=======================================================================
    print('==='*25)
    print('Now running for %s, noise: %s ...'%(target, str(noise_level)))
    #-------------------------------------------------------------------
    X_train = DP.X_train
    Y_train = DP.Y_train
    Y_test = DP.Y_test
    #-------------------------------------------------------------------
    TP = TrainingProcess(X_train, Y_train, n_test, dropout, n_run, target)
    TP.consistent_training()
    kernels = TP.get_kernels('ConsisTrain')
    Y_pred = Prediction(TP, n_train, n_test, kernels)
    #-------------------------------------------------------------------
    print('>> Evaluation ...')
    performance = PerformanceEvaluation(Y_test, Y_pred)
    #-------------------------------------------------------------------
    print('>> Prediction done! Write to files ...')
    SaveResults(Y_train, Y_test, Y_pred, performance, result_dir, target)
    print('==='*25)
#===============================================================================
#===============================================================================