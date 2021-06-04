#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:28:54 2021

@author: ccm
"""
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


#Define the class for Logistic Regression

class LogReg :
    
    def __init__(self ,  Features , Responds):
        scaler = StandardScaler()
        self.Features = scaler.fit_transform(Features)
        self.Responds = Responds
        self.model= LogisticRegression()
        self.model.fit(self.Features , self.Responds[1:])
        

#Define Function for one-step prediction from the Logistic Regression Model

    def prediction (self,x):
        return(self.model.predict(x))
    
class KSVM :
    
    def __init__(self ,  Features , Responds):
        scaler = StandardScaler()
        self.Features = scaler.fit_transform(Features)
        self.Responds = Responds
        self.model= SVC(kernel = 'rbf', random_state = 0)
        self.model.fit(self.Features , self.Responds[1:])

        
        

#Define Function for one-step prediction from the Logistic Regression Model

    def prediction (self,x):
        return(self.model.predict(x))
    
    
    

        
