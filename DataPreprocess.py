#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 20:22:55 2021

@author: ccm
"""

import pandas as pd
import numpy as np

class DataPreprocess : 
    
    def __init__(self , data):
        self.data = data
        self.open = self.data['Open']
        self.close = self.data['Close']
        self.ret = (self.close - self.open)/ self.open
        self.log_ret = self.close/self.open
    
 # Define Function for getting an array of return

    def ret_arr(self):
        return(np.array(self.ret))
    
# Define Function for getting an array of Log_ret

    def log_ret_arr(self):
        return(np.array(self.log_ret))
    
# Define Function for getting an array of Opening Price

    def open_arr(self):
        return (np.array([self.open]))

# Define Function for getting an array of Closing Price

    def close_arr(self):
        return (np.array([self.close]))

# Define Fucntion For getting the indicator
    def indicator(self):
        index = np.zeros(len(self.ret))
        for i in range(len(self.ret)):
            a = self.ret[i]
            if (a > 0):
             index[i] = 1
            if (a  < 0):
             index[i] = 0
            
        return index
    
# Define Function For getting the moving windows of data
    def moving_window(self, start , width , length):
        if (start + width >=length):
            print ("Window out of bound!")
            return 0
        else : 
            return (np.array(self.ret[start , start + width-1]))
        

        


    
