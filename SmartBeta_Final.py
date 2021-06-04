#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:34:20 2021

@author: ccm
"""
import numpy as np
import pandas as pd
import yfinance as yf
import TimeSeries
import AI_Classifier
import PCA_KalmanFilter
import DataPreprocess
import datetime
import math
from sklearn.preprocessing import StandardScaler
import Performance
import sklearn.metrics
import matplotlib.pyplot as plt

# Define the number of stock we want to invest in each time

N = 10

# Define the number of yaer for training up the model

year = 10

# Define the number of look back days to get the PCs

PC_lookback_length = 6

# Define the Principle Amount and Daily_PnL

Principle = np.array([100000])

Daily_PnL = np.array([])

# Define an array for storing the daily profit and Loss

Daily_Profit = np.array([])

# Lodaing all the data we need 

parameters = ['12/31/2016', '12/31/2017','12/31/2018','12/31/2019','12/31/2020',
              datetime.date.today().strftime('%m/%d/%Y')]

# Setting the flag to switch the back test year

flag = 0

# Define the array storing the performance of my classifier

accuracy = np.zeros(len(parameters)-1)

while (flag < len(parameters)-1) :
    
# Adjust the info we use
    
  col_name = parameters[flag]
  date = datetime.datetime.strptime(parameters[flag], '%m/%d/%Y')
  timedelta_1d = datetime.timedelta(days=1)

  start1 = datetime.date(date.year-year, date.month, date.day) + timedelta_1d
  end1 = date

  start2 = datetime.date(date.year - 1, date.month, date.day) + timedelta_1d
  end2 = date

  start3 = date + timedelta_1d
  end3 = datetime.datetime.strptime(parameters[flag+1], '%m/%d/%Y')-timedelta_1d


# Part 0 : Setting the hyperparameter

# Reading the Sticker from the file
  data_df = pd.read_csv('NASDAQ100comp_v2.csv' , header=0 , na_filter=True)

  name = data_df[col_name].drop_duplicates().dropna()

  name = name.reset_index(drop=True)

# Determine the numbers of stock 

  number_of_stock = len(name)

## Part 1 : Train up the Garch model and the Linear Regression Betweeb Indicator and R/sigma^2

# Define the datetime to indicate the lookback period for Index

# Downloading the Index_price from web 
  Index_Price_p1 = yf.download('^IXIC', 
                          start= start1, 
                          end= end1, 
                          progress=False,
                          auto_adjust = True)

# Store the total length of Index_price in part 1

  Index_timelength_p1 = Index_Price_p1.index.shape[0]
  
# Define the max_length and Max_index for aligning the transaction record in the stock price data

  max_day = -1
  max_index = -1
  


# Downloading the Stock_price from web and prepare a matrix for storing the daily return of stock for initialize the b_t and A_t in Kalman Filter Model

  Stock_price_collection_p1 = {}

  for i in range (number_of_stock):
    
       Stock_price_collection_p1[i] = yf.download(name[i], 
                                start = start2, 
                                end = end2, 
                                progress=False,
                                auto_adjust = True)
  for i in range (number_of_stock) :
      
        if (Stock_price_collection_p1[i].index.shape[0] > max_day):
            max_day = Stock_price_collection_p1[i].index.shape[0]
            max_index = i
            
  for i in range (number_of_stock) :
            Stock_price_collection_p1[i] = Stock_price_collection_p1[i].reindex(Stock_price_collection_p1[max_index].index)            
            Stock_price_collection_p1[i] = Stock_price_collection_p1[i].fillna(method = 'ffill' , axis = 0)
            
      
      
  for i in range (number_of_stock):
    
        Stock_price_p1 = Stock_price_collection_p1[i]
        
        data_stock_p1 = DataPreprocess.DataPreprocess(Stock_price_p1)
      
        if (i == 0) :
              Stock_ret_p1 = np.matrix(data_stock_p1.ret_arr()).T

        else :
              Stock_ret_p1 = np.concatenate((Stock_ret_p1,np.matrix(data_stock_p1.ret_arr()).T),axis=1)
              
# Getting the set of first PCs

  sample_stock = Stock_ret_p1
  sample_index = DataPreprocess.DataPreprocess(Index_Price_p1.reindex(Stock_price_p1.index)).ret_arr()
  sample_size = sample_stock.shape[0]
  i =0
  while(i + (PC_lookback_length - 1) < sample_size):
        sample_stock_return = sample_stock[i : i + PC_lookback_length]
        sample_index_return = sample_index[i : i + PC_lookback_length]
        PCs = PCA_KalmanFilter.PCloading(sample_stock_return)
        corr_corf = np.corrcoef(sample_stock_return.T , sample_index_return.T)[:-1,-1]
        result = PCs * corr_corf
        if(np.sum(result>0) / len(result[0]) < 0.5):
            PCs = -1 * PCs
        
        if (i == 0):
            sample_PCs = PCs
        else : 
            sample_PCs = np.concatenate((sample_PCs , PCs) , axis =0)
            
        i+=1

## Part 2.1 : 

# Downloading the Index price and Stock_price from web for back testing

# Downloading the Index Price

  Index_Price_p2 = yf.download('^IXIC', 
                          start= start3, 
                          end= end3, 
                          progress=False,
                          auto_adjust = True)
  
  max_day = Index_Price_p2.index.shape[0]
  
  if (flag ==0):
        timeindex = Index_Price_p2.index
  else:
        timeindex = timeindex.append(Index_Price_p2.index[1:])


# Downloading the component Stock Price

  Stock_price_collection_p2 = {}

  for i in range (number_of_stock):
    
        Stock_price_collection_p2[i] = yf.download(name[i], 
                                start = start3, 
                                end = end3, 
                                progress=False,
                                auto_adjust = True)
          
  for i in range (number_of_stock) :
      
        if (Stock_price_collection_p2[i].index.shape[0] != max_day):
            Stock_price_collection_p2[i] = Stock_price_collection_p2[i].reindex(Index_Price_p2.index)
            
  for i in range (number_of_stock) :            
        Stock_price_collection_p2[i] = Stock_price_collection_p2[i].fillna(method = 'ffill' , axis = 0)

# Get the daily return

  Index_ret_p2 = DataPreprocess.DataPreprocess(Index_Price_p2).ret_arr()

# Merging the index price from part 1 and part 2

  Index_Price = pd.concat([Index_Price_p1 , Index_Price_p2] , axis =0)

  Index_data = DataPreprocess.DataPreprocess(Index_Price)

  Index_ret = Index_data.ret_arr()
      
# Determine the numbers of day in backtest

  number_of_day = max_day
      
# Get the Daily Return of each Stock
      
  for i in range (number_of_stock):
    
        Stock_price_p2 = Stock_price_collection_p2[i]
        
        data_stock_p2 = DataPreprocess.DataPreprocess(Stock_price_p2)
      
        if (i == 0) :
              Stock_ret_p2 = np.matrix(data_stock_p2.ret_arr()).T

        else :
              Stock_ret_p2 = np.concatenate((Stock_ret_p2,np.matrix(data_stock_p2.ret_arr()).T),axis=1)
            
# Get the Open Price of each Stock

  for i in range (number_of_stock):
    
        Stock_price_p2 = Stock_price_collection_p2[i]
        
        data_stock_p2 = DataPreprocess.DataPreprocess(Stock_price_p2)
      
        if (i == 0) :
            Stock_Open = np.matrix(data_stock_p2.open_arr()).T
        else :
            Stock_Open = np.concatenate((Stock_Open,np.matrix(data_stock_p2.open_arr()).T),axis=1)

# Get the Close Price of each Stock

  for i in range (number_of_stock):
    
        Stock_price_p2 = Stock_price_collection_p2[i]
        
        data_stock_p2 = DataPreprocess.DataPreprocess(Stock_price_p2)
      
        if (i == 0) :
            Stock_Close = np.matrix(data_stock_p2.close_arr()).T

        else :
            Stock_Close = np.concatenate((Stock_Close,np.matrix(data_stock_p2.close_arr()).T),axis=1)

# Concantenate the last 4 stock return record from last year into the back test record for the getting PCs

  Stock_ret = np.concatenate((Stock_ret_p1[-(PC_lookback_length-1):] , Stock_ret_p2) , axis =0)

# Part 2.2 Back test

# initializing the parameter

  d = 0

  PCs_old = sample_PCs[-1]

  for j in range(len(Stock_price_p1) - (PC_lookback_length -1 )):
# Storing the transition_error
      if (j == sample_PCs.shape[1] -1):
          transition_error_series = np.matrix(np.zeros(sample_PCs.shape[1]))
      elif (j >= sample_PCs.shape[1]) :
          PC = sample_PCs[:j+1]
          b_t , A_t = PCA_KalmanFilter.VectorARmodel(PC)
          transition_error = np.matrix(PC[-1]) - (b_t + np.matrix(PC[-2]) @ A_t)
          transition_error_series = np.concatenate((transition_error_series , transition_error) , axis =0)
      
      
# Storing the observed error    
      y_t = sample_PCs[j] @ Stock_ret_p1[j].T

      C_t = y_t[0,0] * np.identity(number_of_stock)

      observed_error = Stock_ret_p1[j] - sample_PCs[j] @ C_t
    
      if (j ==0) :
        
          observed_error_series = observed_error
        
      else : 
        
          observed_error_series = np.concatenate((observed_error_series , observed_error) , axis =0)

  post_x = np.matrix(sample_PCs[-1]).T

  post_cov = np.cov(np.matrix(sample_PCs).T)

  Indicator = np.zeros(number_of_day - 1)

  while (d < number_of_day - 1):
    
# Train the model of Grach and Classifier fo new info

      Index_data_backtest = Index_Price.iloc[d : d+Index_timelength_p1]
    
      Index_dp_backtest = DataPreprocess.DataPreprocess(Index_data_backtest)
    
      Index_ret_backtest = Index_dp_backtest.ret_arr()
    
      ts_Index_ret_backtest = TimeSeries.Timeseries(Index_ret_backtest)

      garch_w , vol_backtest = ts_Index_ret_backtest.garchmodel(1,1)
    
      X1 = np.matrix(Index_ret_backtest)

      X2 = np.matrix(vol_backtest)

      Features = np.concatenate((X1.T,X2.T) , axis =1)

      sc = StandardScaler()
 
      Features = sc.fit_transform(Features)

      Responds = Index_dp_backtest.indicator() 
    
      new_responds = (1 if Index_ret_p2[d]>0 else 0)
    
      Responds = np.append(Responds,new_responds) 

      ML_Index_ret = AI_Classifier.LogReg(Features, Responds)
    
# Getting the Index return for prediction of Indicator

      Index_r = Index_ret_p2[d]

# Estimating the today volatility from garch model

      prev_vol = vol_backtest[-1]

      new_vol = garch_w[0] + garch_w[1] * (Index_r)**2 + garch_w[2] * prev_vol
    
# Predicting the indicator about the index movement from Logistic Regression Model

      Indicator[d] = ML_Index_ret.prediction(sc.transform([[Index_r , new_vol]]))
    
# Getting lookback period  Stock Return Series 

      Stock_r = Stock_ret[d : d+PC_lookback_length]
    
# Getting the PCs for Finding C_t
   
      PCs_t = PCA_KalmanFilter.PCloading(Stock_r)
 
# Adjusting the PC loadings by the testing whether the direction of loading align with majority of stock and marekt Corr      
        
      sample_index_test = np.append(sample_index ,Index_ret_p2[:d+1] , axis =0)
      sample_size = sample_index_test.shape[0]
      i = sample_size - PC_lookback_length
      sample_index_return = sample_index_test[i : i + PC_lookback_length]
      corr_corf = np.corrcoef(Stock_r.T , sample_index_return.T)[:-1,-1]
      result = PCs * corr_corf
      if(np.sum(result>0) / len(result[0]) < 0.5):
            PCs_t = -1 * PCs_t
      
      sample_PCs = np.concatenate((sample_PCs , PCs_t) , axis =0)
      
# Update the info on PCs for next Kalman filter Gain
      
#      post_x = np.matrix(PCs_t).T
      
# Estimating b_t and A_t       

      b_t , A_t = PCA_KalmanFilter.VectorARmodel(sample_PCs)
    
# Estimating C_t for kalman Filter
    
      y_t = PCs_t @ Stock_r[-1].T

      C_t = y_t[0,0] * np.identity(number_of_stock)

# Finding the Filtered PCs

      transition_error = np.matrix(PCs_t) - (b_t + np.matrix(PCs_old) @ A_t)

      transition_error_series = np.concatenate((transition_error_series , transition_error) , axis =0) 
    
      Q_t = PCA_KalmanFilter.TransitionErrorVariance(transition_error_series.T)
    
      observed_error = Stock_r[-1] - PCs_t @ C_t
    
      observed_error_series = np.concatenate((observed_error_series , observed_error) , axis =0)
    
      R_t = PCA_KalmanFilter.MeasurementErrorVariance(observed_error_series.T)
    
      v_t = C_t @ (PCs_t.T - post_x) + observed_error.T
    
      V_t = PCA_KalmanFilter.V_t(C_t , post_cov , R_t)
    
      K_t = PCA_KalmanFilter.KalmanGain(A_t.T , post_cov , C_t , V_t)
    
# Compute the posterior Prediction on next x , cov matrix
    
      post_x = PCA_KalmanFilter.post_pred_x(b_t, A_t.T, post_x, K_t , v_t)
    
      post_cov = PCA_KalmanFilter.post_pred_Var(A_t.T , post_cov , K_t , C_t , Q_t)
    
# Sorting the Filtered PCs

      pred_PC = pd.DataFrame(post_x , columns = ['Stock'])
      pred_PC.index = name
      Sort_pred_PC = pred_PC.sort_values(by=['Stock'],axis = 0 , ascending=False)
      
# Saving the PCs_t to PCs_old

      PCs_old = PCs_t
    
# Picking the stock from The number of Stock by our defined starting position and Number of stocks
    
      target_list_long = list()
      target_list_short = list()
 
      for a in range(N) : 
          target_list_long.append(Sort_pred_PC.index[a])
          target_list_short.append(Sort_pred_PC.index[-a-1])
    
# Getting the d+1 open price and close    

      Next_Open = Stock_Open[d+1]
    
      Next_Open_df = pd.DataFrame(Next_Open.T)
    
      Next_Open_df.index = name
    
      Next_Close = Stock_Close[d+1]
    
      Next_Close_df = pd.DataFrame(Next_Close.T)
    
      Next_Close_df.index = name
    
# Picking the target return and Filtered PCs 

      print("Now calculating PnL")
      
      target_open_long = np.array([])
    
      target_close_long = np.array([])
      
      target_pred_pc_long = np.array([])
    
      target_open_short = np.array([])
    
      target_close_short = np.array([])
      
      target_pred_pc_short = np.array([])

      for c in range(N):
          ticker = target_list_long[c]
          target_open_long = np.append(target_open_long , Next_Open_df.loc[ticker][0])
          target_close_long = np.append(target_close_long , Next_Close_df.loc[ticker][0])
          target_pred_pc_long = np.append(target_pred_pc_long , pred_PC.loc[ticker][0])
          
      for c in range(N):
          ticker = target_list_short[c]
          target_open_short = np.append(target_open_short , Next_Open_df.loc[ticker][0])
          target_close_short = np.append(target_close_short , Next_Close_df.loc[ticker][0])
          target_pred_pc_short = np.append(target_pred_pc_short , (pred_PC.loc[ticker][0] if pred_PC.loc[ticker][0] < 0 else 0 ) ) 
          
# Adjust the sign of PC according to the  indicattor in order to adjust the weight (Determining Long /Short)

      target_pred_pc_long = (target_pred_pc_long if Indicator[d] ==1 else -1 * target_pred_pc_long)
      target_pred_pc_short = (target_pred_pc_short if Indicator[d] ==1 else -1 * target_pred_pc_short)
          
# counting the weight

    
# Computing the number of share we buy for next day and also the price we brought

      weight_long = target_pred_pc_long / sum(abs(target_pred_pc_long))
      weight_short = ( target_pred_pc_short / sum(abs(target_pred_pc_short)) if sum(abs(target_pred_pc_short)) >0 else np.zeros(N))
 
      number_of_share_brought_long = np.array([])
      number_of_share_brought_short = np.array([])
      
      asset_long_weight = sum(abs(target_pred_pc_long)) / (sum(abs(target_pred_pc_long)) + sum(abs(target_pred_pc_short)))
      asset_short_weight = 1- asset_long_weight
    
      for i in range(N):
        w = weight_long[i] 
        open_price = target_open_long[i]
        number_of_share_brought_long= np.append(number_of_share_brought_long , math.floor(Principle[-1] * asset_long_weight * w / open_price))
        
      for i in range(N):
        w = weight_short[i] 
        open_price = target_open_short[i]
        number_of_share_brought_short= np.append(number_of_share_brought_short , math.ceil(Principle[-1] * asset_short_weight * w / open_price))
        
      
# Computing the Daily Profit and Update the Principle

      Daily_Profit = np.append(Daily_Profit ,sum(number_of_share_brought_long*(target_close_long - target_open_long) + number_of_share_brought_short*(target_close_short - target_open_short)))
    
      Principle = np.append(Principle ,(Principle[-1] + Daily_Profit[-1]))
    
      d+=1
      
  accuracy[flag] = sklearn.metrics.accuracy_score(Index_ret_p2[1:]>0 , Indicator)
      
  flag +=1
    
# Prepare a Df


  Principle_df = pd.DataFrame(Principle)
  
  Principle_df.index = timeindex
  
  Principle_df.columns = ['Strat']

  Principle_df.plot(xlabel = "Date" , ylabel = "NAV")
  
  plot = Principle_df.plot(xlabel = "Date" , ylabel = "NAV")
  
  
#Performance

metrics = Performance.Measures(Principle)

print('CR : ' , metrics.Cumulative_Return())
print('AR : ' , metrics.Annualized_GM())
print('Vol : ', metrics.Annualized_Vol() )
print('ASR : ' , metrics.Annualized_Sharpe())
print('Sortino Ratio : ' ,metrics.Sortino_Ratio())
print('MDD : ' , metrics.MaxDrawDown())
print('Calmar Ratio : ', metrics.CalmarRatio())

measure = {'CR : ' , metrics.Cumulative_Return(),
           'AR : ' , metrics.Annualized_GM(),
           'Vol : ', metrics.Annualized_Vol() ,
           'ASR : ' , metrics.Annualized_Sharpe() ,
           'Sortino Ratio : ' ,metrics.Sortino_Ratio(),
           'MDD : ' , metrics.MaxDrawDown(),
           'Calmar Ratio : ', metrics.CalmarRatio()}

measure_df = pd.DataFrame(data = measure)

measure_df.to_csv("Measure_v2.csv")

# Export CSV and graph

Principle_df.to_csv('Daily_NAV_v2.csv')
