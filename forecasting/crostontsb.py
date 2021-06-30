import numpy as np
import pandas as pd
from forecasting.utils import forecast_df, calculate_mae
from pandas.tseries.offsets import DateOffset

VAL_COL = 'ship_qty'
TSB = 'tsb'
CROSTON = 'croston'

#fcast_period=3
#alpha=0.4
#beta=0.4

def Croston(ts, arg, alpha=0.4):
    print("Running Croston model. start_date: {}, last_date: {}".format(arg.train_data.Date.iloc[0],
                                                                     arg.train_data.Date.iloc[-1]))
    extra_periods = arg.hp
    # transform the input into a numpy array
    array_df = np.array(ts)

    # historical period length
    cols = len(array_df)
    # append np.nan into the demand array to cover future periods
    array_df = np.append(array_df,[np.nan]*extra_periods)
    
    # demand level (a), periodicity (p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods),np.nan)
    # periods since last demand observation
    q = 1
    
    # initialization
    first_occurence = np.argmax(array_df[:cols]>0)
    a[0] = array_df[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0]/p[0]
    
    # create all the t+1 forecasts
    # elapsed time since previous demand occurrence (q)
    for t in range(0,cols):        
        if array_df[t] > 0:
            a[t+1] = alpha*array_df[t] + (1-alpha)*a[t] 
            p[t+1] = alpha*q + (1-alpha)*p[t]
            f[t+1] = a[t+1]/p[t+1]
            q = 1           
        else:
            a[t+1] = a[t]
            p[t+1] = p[t]
            f[t+1] = f[t]
            q += 1
       
    # future forecast 
    a[cols+1:cols+extra_periods] = a[cols]
    p[cols+1:cols+extra_periods] = p[cols]
    f[cols+1:cols+extra_periods] = f[cols]
                      
    df_train = pd.DataFrame.from_dict({"Demand":array_df,"Forecast":f,"Period":p,"Level":a,"Error":array_df-f})
    
    return df_train




def Croston_TSB(ts, arg, alpha=0.4,beta=0.4):
    
    print("Running Croston-TSB model. start_date: {}, last_date: {}".format(arg.train_data.Date.iloc[0],
                                                                            arg.train_data.Date.iloc[-1]))
    extra_periods = arg.hp
    # transform the input into a numpy array
    array_df = np.array(ts)
    # historical period length
    cols = len(array_df)
    # append np.nan into the demand array to cover future periods
    array_df = np.append(array_df,[np.nan]*extra_periods)
    
    # level (a), probability(p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods),np.nan)
    
    # initialization
    first_occurence = np.argmax(array_df[:cols]>0)
    a[0] = array_df[first_occurence]
    p[0] = 1/(1 + first_occurence)
    f[0] = p[0]*a[0]
                 
    # create all the t+1 forecasts
    for t in range(0, cols):
        if array_df[t] > 0:
            a[t+1] = alpha*array_df[t] + (1-alpha)*a[t] 
            p[t+1] = beta*(1) + (1-beta)*p[t]  
        else:
            a[t+1] = a[t]
            p[t+1] = (1-beta)*p[t]       
        f[t+1] = p[t+1]*a[t+1]
        
    # future forecast
    a[cols+1:cols+extra_periods] = a[cols]
    p[cols+1:cols+extra_periods] = p[cols]
    f[cols+1:cols+extra_periods] = f[cols]
                      
    df_train = pd.DataFrame.from_dict({"Demand": array_df, "Forecast": f, "Period": p, "Level": a, "Error": array_df-f})
    
    return df_train





def croston_function(arg):
    y_pred = np.round_(Croston(arg.train_data[VAL_COL], arg))['Forecast'].iloc[-arg.hp:]
    last_training_date =arg.train_data.tail(1).index + DateOffset(months=1)
    dt = pd.date_range(start=last_training_date[0], periods=arg.hp, freq="MS")
    y_pred.index = dt
    df_forecast = forecast_df(arg.parameters, y_pred, None, None, CROSTON)
    return df_forecast



#CROSTON TSB: roll forward cross validation
def tsb_function(arg):
    y_pred = np.round_(Croston_TSB(arg.train_data[VAL_COL], arg))['Forecast'].iloc[-arg.hp:]
    last_training_date =arg.train_data.tail(1).index + DateOffset(months=1)
    dt = pd.date_range(start=last_training_date[0], periods=arg.hp, freq="MS")
    y_pred.index = dt
    df_forecast = forecast_df(arg.parameters, y_pred, None, None, TSB)
    return df_forecast

#CROSTON TSB: roll forward cross validation
# def tsb_roll_forward_cv(arg):
#     # y_pred=[]
#     # index_min = len(arg.train_data)
#     # index_max = arg.parameters.holdout_periods[0] + index_min
#     # for train_index in range(index_min, index_max):
#     #     training_df = arg.train_data.iloc[:train_index]
#         #predict 1 day ahead and save y_pred
#     y_pred = (round(Croston_TSB(arg.train_data[VAL_COL], arg), 0)['Forecast'].iloc[-1])
#     df_fcast = pd.DataFrame()
#     mae, av_mae = calculate_mae(y_pred, arg.val_data[VAL_COL], TSB)
#     df_forecast = forecast_df(arg.parameters, y_pred, None, None, arg.val_data[VAL_COL], mae,
#                                    av_mae, TSB)
#     return df_forecast