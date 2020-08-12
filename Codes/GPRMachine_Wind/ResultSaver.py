#!/usr/bin/python  
# -*- coding: utf-8 -*- 
#===============================================================================
# Date      : Feb/28/2020
# Author    : XHHAO
# Annotation: This file is for saving prediction results.
#===============================================================================
#===============================================================================
def SaveResults(Y_train, Y_test, Y_pred, performance, repeat, result_dir, target):
    #===========================================================================
    # --------------------------------- X -------------------------------------#
    #===========================================================================
    YTr = list(Y_train)
    YTe = list(Y_test)
    YPr = list(Y_pred)
    mae_x = performance[0]
    rmse_x = performance[1]
    pcc_x = performance[2]
    result_name_x = result_dir + '/' + target + '_' + str(mae_x) + '_' +\
                    str(rmse_x) + '_' + str(pcc_x) + '_' + str(repeat) + '.result'
    result_file_x = open(result_name_x, 'w')
    for y in YTr:
        result_file_x.write(str(y) + '\t' + '\t' + '\n')
    for y, ypr in zip(YTe, YPr):
        result_file_x.write(str(y) + '\t' + str(ypr) + '\n')
    result_file_x.close()
#===============================================================================
#===============================================================================