#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 00:49:19 2021

@author: ccm
"""

import numpy as np
from sklearn.decomposition import PCA 
from statsmodels.tsa.vector_ar.var_model import VAR
from numpy.linalg import inv

    
# Define the function for getting the 1st PC loadings

def PCloading(data) : 
        pca = PCA(n_components=1 , random_state=0)
        pca.fit(data)
        return(pca.components_ * np.sqrt(pca.explained_variance_))
    
# Define the function for obtaining the transition offset, the transition Matrix and Variance of state errors
def VectorARmodel(data) : 
        number_of_features = data.shape[1]
        model = VAR(data)
        model_fitted = model.fit(maxlags = 1 , trend = 'n')
        return (np.zeros([1,number_of_features]) , model_fitted.params[0:] )
    
# Define the function for outputing the different between actual y (y_t) and posterior predict y (y_t|t-1) as v_t

def nu(y , post_y) : 
        return (y-post_y)
    
# Define the function for finding Variance on Measurement Error

def MeasurementErrorVariance(measure_error):
        return (np.cov(measure_error))
    
# Define the function for finding Variance on Transition Error

def TransitionErrorVariance(transition_error):
        return (np.cov(transition_error))
   
# Define the function for Variance on difference in prediction (the different between actual y (y_t) and posterior predict y (y_t|t-1)) as V_t

def V_t(C_t , post_Var , R_t):
        return (C_t @ post_Var @ C_t.T + R_t)
    
# Define the Function for Kalman Gain as K_t

def KalmanGain(A_t , post_Var , C_t , V_t):
        return (A_t @ post_Var @ C_t.T @ inv(V_t))
    
# Define the Function for one-step posterior prediction on x

def post_pred_x(b_t , A_t , posterior_x , K_t , v_t):
        return (np.matrix(b_t).T + np.dot(A_t , posterior_x) + np.dot(K_t,v_t))
    
# Define the Function for one-step posterior prediction on Variance of x

def post_pred_Var(A_t , posterior_Var , K_t , C_t , Q_t):
       L_t = A_t - K_t @ C_t
       return (A_t @ posterior_Var @ L_t.T + Q_t)
   
# Define the Function for one-step posterior prediction on y

def post_pred_y(d_t , C_t , post_x):
        return (np.matrix(d_t).T + np.dot(C_t,post_x))
        
    
    
        
        

   
    