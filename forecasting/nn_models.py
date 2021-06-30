from datetime import datetime
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from data_manager.table_names import *
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from forecasting.utils import get_r_df, fit_df, fitted_df, forecast_df, calculate_mae, \
    pi_function

ts = robjects.r('ts')
pandas2ri.activate()
stats = importr('stats')
forecast = importr('forecast')
set_seed = robjects.r('set.seed')

VAL_COL = 'ship_qty'
INT_VAR = 'order_qty'
NN_Xreg = 'nn_xreg'
NN = 'nn'


def nn_xreg(arg):
    #        NN model with regressors
    # Add key from the metric file
    ##arg.metrics.add(key, datetime.now())
    print("Running NNXREG. training from: {}, till: {}".format(arg.train_data.Date.iloc[0],
                                                               arg.train_data.Date.iloc[-1], arg.hp))  # arg.hp
    #   train the model
    model_nnxreg, df_train = nn_xreg_training(arg)

    #        Runs if variables have orders as a relevant variable along with other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) > 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor
        forecast_output = forecast.forecast(model_nnxreg, len(Xreg), xreg=Xreg)
        forecast_output_dict = dict(zip(forecast_output.names, list(forecast_output)))
        r_forecast_order = get_r_df(forecast_output_dict['mean'], None, None, arg.train_data.Date.max(),
                                    arg.parameters.tsfreq)
        r_forecast_order["mean"] = np.round_(r_forecast_order["mean"])
        r_forecast_order['upper'], r_forecast_order['lower'] = pi_function(arg.train_data[VAL_COL],
                                                                           df_train[forecast_value],
                                                                           r_forecast_order["mean"],
                                                                           p=len(arg.train_var.columns))

        #            train_data_modified = arg.train_data
        #            train_var_modified = arg.train_var
        arg.train_data.loc[-1] = [arg.train_data["type"][0], arg.train_data["Part_number"][0],
                                  arg.train_data["Location"][0], r_forecast_order["mean"][0],
                                  r_forecast_order["Date"][0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)
        arg.train_var.loc[-1] = arg.test_var.iloc[0]
        arg.train_var = arg.train_var.drop(columns=INT_VAR)  # c remove orders from the variables mix in train
        model_nnxreg, df_train = nn_xreg_training(arg)
        Xreg = arg.test_var.iloc[1:, :]
        Xreg = Xreg.drop(columns=INT_VAR)  # c remove orders from the variables mix in val
        forecast_output = forecast.forecast(model_nnxreg, len(Xreg), xreg=Xreg.values)
        forecast_output_dict = dict(zip(forecast_output.names, list(forecast_output)))
        r_forecast_ext_var = get_r_df(forecast_output_dict['mean'], None, None, arg.train_data.Date.max(),
                                      arg.parameters.tsfreq)
        r_forecast_ext_var['mean'] = np.round_(r_forecast_ext_var['mean'])
        r_forecast_ext_var['upper'], r_forecast_ext_var['lower'] = pi_function(arg.train_data[VAL_COL],
                                                                               df_train[forecast_value],
                                                                               r_forecast_ext_var["mean"],
                                                                               p=len(arg.train_var.columns))
        r_forecast_ext_var = r_forecast_order.append(r_forecast_ext_var)

        df_forecast_ext_var = forecast_df(arg.parameters, r_forecast_ext_var['mean'], r_forecast_ext_var['upper'], r_forecast_ext_var['lower'], NN_Xreg)

        arg.hp = arg.hp - len(Xreg) - 1  # ()-1 for orders too)
        if arg.hp != 0:
            df_forecast_ext_var[forecast_value] = np.where(df_forecast_ext_var[forecast_value] < 0, 0,
                                                     df_forecast_ext_var[forecast_value])
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": r_forecast_ext_var["mean"],
                 "Date": r_forecast_ext_var.index})

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts = nn(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
        else:
            df_forecast = df_forecast_ext_var





    #        Runs if variables have orders as a relevant variable along with other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) == 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor
        forecast_output = forecast.forecast(model_nnxreg, len(Xreg), xreg=Xreg.values)
        forecast_output_dict = dict(zip(forecast_output.names, list(forecast_output)))
        r_forecast_order = get_r_df(forecast_output_dict['mean'], None, None, arg.train_data.Date.max(),
                                    arg.parameters.tsfreq)
        r_forecast_order['upper'], r_forecast_order['lower'] = pi_function(arg.train_data[VAL_COL],
                                                                           df_train[forecast_value],
                                                                           r_forecast_order["mean"],
                                                                           p=len(arg.train_var.columns))
        r_forecast_order['mean'] = np.round_(r_forecast_order['mean'], 0)
        # mae, av_mae = calculate_mae(r_forecast_order['mean'], arg.val_data[VAL_COL][0], NN_Xreg)
        df_forecast_order = forecast_df(arg.parameters, r_forecast_order['mean'], r_forecast_order["upper"],
                                        r_forecast_order["lower"], NN_Xreg)

        #            modify data for pure time series models
        arg.train_data.loc[-1] = [arg.train_data["type"][0], arg.train_data["Part_number"][0],
                                  arg.train_data["Location"][0], r_forecast_order["mean"][0],
                                  r_forecast_order["Date"][0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)
        #            kick off pure time series module as no other variable would be used to forecast
        arg.hp = arg.hp - 1
        df_train_pts_int, df_forecast_pts = nn(arg)
        df_forecast = df_forecast_order.append(df_forecast_pts)

    #        runs with just external variables
    if (INT_VAR not in arg.var_names):
        Xreg = arg.test_var
        forecast_output = forecast.forecast(model_nnxreg, len(Xreg), xreg=Xreg.values)
        forecast_output_dict = dict(zip(forecast_output.names, list(forecast_output)))
        r_forecast = get_r_df(forecast_output_dict['mean'], None, None, arg.train_data.Date.max(),
                              arg.parameters.tsfreq)
        r_forecast['upper'], r_forecast['lower'] = pi_function(arg.train_data[VAL_COL],
                                                               df_train[forecast_value],
                                                               r_forecast["mean"],
                                                               p=len(arg.train_var.columns))
        r_forecast['mean'] = np.round_(r_forecast['mean'], 0)

        df_forecast_ext_var = forecast_df(arg.parameters, r_forecast['mean'], r_forecast['upper'], r_forecast['lower'], NN_Xreg)

        arg.hp = arg.hp-len(Xreg)

        if arg.hp != 0:
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": r_forecast['mean'],
                 "Date": df_forecast_ext_var.index})
            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts= nn(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
        else:
            df_forecast = df_forecast_ext_var

    return df_train, df_forecast


def nn_xreg_training(arg):
    #    training module for neural network with external variables
    rdata = ts(arg.train_data[VAL_COL], frequency=arg.parameters.tsfreq)
    set_seed(1000)
    fit = forecast.nnetar(rdata, xreg=arg.train_var.values)
    fit_dict = dict(zip(fit.names, list(fit)))
    fitted_values = fit_dict['fitted']
    fitted = fitted_df(fitted_values, arg.train_data['Date'].min(), arg.parameters.tsfreq)
    fitted['fitted_values'] = np.round_(fitted['fitted_values'], 0)
    mae = calculate_mae(fitted['fitted_values'], arg.train_data[VAL_COL].values, NN_Xreg)
    # upper, lower = pi_function(arg.train_data[VAL_COL], fitted['fitted_values'], fitted['fitted_values'], p = len(arg.train_var.columns))
    df_train = fit_df(fitted['fitted_values'], arg.train_data[VAL_COL].values, mae, NN_Xreg)
    return fit, df_train


def nn(arg):
    #        NN model without Regressors
    print("Running NN. start_date: {}, last_date: {}, period: {}".format(arg.train_data.Date.iloc[0],
                                                                         arg.train_data.Date.iloc[-1], arg.hp))
    rdata = ts(arg.train_data[VAL_COL], frequency=arg.parameters.tsfreq)
    set_seed(1000)
    fit = forecast.nnetar(rdata)
    fit_dict = dict(zip(fit.names, list(fit)))
    fitted_values = fit_dict['fitted']
    fitted = fitted_df(fitted_values, arg.train_data['Date'].min(), arg.parameters.tsfreq)
    fitted['fitted_values'] = np.round_(fitted['fitted_values'], 0)
    mae = calculate_mae(fitted['fitted_values'], arg.train_data[VAL_COL].values, NN)
    df_train = fit_df(fitted['fitted_values'], arg.train_data[VAL_COL].values, mae, NN)

    set_seed(1000)
    forecast_output = forecast.forecast(fit, arg.hp)
    forecast_dict = dict(zip(forecast_output.names, list(forecast_output)))
    r_df = get_r_df(forecast_dict['mean'], None, None, arg.train_data.Date.max(), arg.parameters.tsfreq)
    r_df['mean'] = np.round_(r_df['mean'], 0)
    r_df['upper'], r_df['lower'] = pi_function(arg.train_data[VAL_COL].values, fitted['fitted_values'], r_df['mean'],
                                               p=0)

    df_forecast = forecast_df(arg.parameters, r_df['mean'], r_df['upper'], r_df['lower'], NN)
    return df_train, df_forecast
