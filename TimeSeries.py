#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:24:45 2021

@author: ccm
"""
#Define the class for Time Series Modeling

import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model

class Timeseries :

#Define a constructor for object creation

    def __init__(self , data) :
        self.data = data
                
#Define Function for checking the autocorrelation
       
    def adftest(self,lags):
        test = adfuller(self.data ,  maxlag = lags)
        print ("pvalue : %f " @ test[1])
        print("usedlag : %f" @ test[2])

# Define Function for training the AR model and make a one-step prediction

    def armodel(self,lags):
        model = AutoReg(self.data , lags)
        model_fit = model.fit()
        coef = model_fit.params
        return (coef[0] + coef[1] * self.data[-1])


# Define Function for training the GARCH model return the weight

    def garchmodel(self , p , q) :
        model = arch_model(self.data , mean = 'Constant' , vol = 'GARCH' , p= p , q=q)
        model_fit = model.fit()
        w = np.array(model_fit.params)[1:]
        r_sq = self.data**2
        vol = [0]
        i = 0
        while (i < len(self.data)):
            v = w[0] + w[1] * r_sq[i] + w[2] * vol[i]
            vol = np.concatenate((vol , [v]))
            i = i +1
            
        
        return (w,vol[:-1])
        
        
        
        
        
    
    