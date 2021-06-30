from datetime import datetime
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from forecasting.utils import *

ts = robjects.r('ts')
c =  robjects.r('c')
pandas2ri.activate()
stats = importr('stats')
forecast = importr('forecast')
set_seed = robjects.r('set.seed')

VAL_COL = 'ship_qty'
INT_VAR = 'order_qty'
ARIMA_Xreg = 'arima_xreg'
ARIMA = 'arima'




# TODO add upper and lower confidence intervals

def arima_xreg(arg):

    model_coeff_full = []
    model_order_full = []
    model_order_interim = []
    model_coeff_full_interim = []


    print("Running ARIMA_XREG. training from: {}, till: {}".format(arg.train_data.Date.iloc[0],
                                                                   arg.train_data.Date.iloc[-1]))  # arg.hp
    #   train the model
    model_arima_xreg, df_train, coefficients_dict = arima_xreg_training(arg)

    set_seed(1000)

    #        Runs if variables have orders as a relevant variable along with other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) > 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor
        forecast_output = forecast.forecast(model_arima_xreg, h=len(Xreg), xreg=Xreg.values)
        forecast_output_dict = dict(zip(forecast_output.names, list(forecast_output)))
        r_forecast_order = get_r_df(forecast_output_dict['mean'], None, None, arg.train_data.Date.max(),
                                    arg.parameters.tsfreq)
        arg.train_data.loc[-1] = [arg.train_data["type"][0], arg.train_data["Part_number"][0],
                                  arg.train_data["Location"][0], r_forecast_order["mean"][0],
                                  r_forecast_order["Date"][0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)
        arg.train_var.loc[-1] = arg.test_var.iloc[0]
        arg.train_var.index = arg.train_data.index
        arg.train_var = arg.train_var.drop(columns=INT_VAR)  # remove orders from the variables mix in train
        model_arima_xreg, df_train, coefficients_dict_ext_var = arima_xreg_training(arg)
        Xreg = arg.test_var.iloc[1:, :]
        Xreg = Xreg.drop(columns=INT_VAR)  # c remove orders from the variables mix in val
        forecast_output_ext_var = forecast.forecast(model_arima_xreg, len(Xreg), xreg=Xreg.values)
        forecast_output_dict = dict(zip(forecast_output_ext_var.names, list(forecast_output_ext_var)))
        r_forecast_ext_var = get_r_df(forecast_output_dict['mean'], None, None, arg.train_data.Date.max(),
                                      arg.parameters.tsfreq)
        r_forecast_ext_var['upper'], r_forecast_ext_var['lower'] = pi_function(arg.train_data[VAL_COL].values, df_train[forecast_value],
                                                   r_forecast_ext_var['mean'], p= len(arg.train_var.columns))
        r_forecast = r_forecast_order.append(r_forecast_ext_var)
        df_forecast_ext_var = forecast_df(arg.parameters, r_forecast_ext_var['mean'], r_forecast_ext_var['upper'], r_forecast_ext_var['lower'], ARIMA_Xreg)

        arg.hp  = arg.hp-len(Xreg)-1 # ()-1 for orders too)
        if arg.hp != 0:
            r_forecast["mean"] = np.where(r_forecast["mean"] <0, 0, r_forecast["mean"] )
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": r_forecast_ext_var['mean'],
                 "Date": r_forecast_ext_var["Date"]})

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts, model_param_df = arima(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
        else:
            df_forecast = df_forecast_ext_var
        # output model parameters
        model_order_order = (forecast_output[0])
        model_order_ext_var = (forecast_output_ext_var[0])
        model_order_interim.append(model_order_order)
        model_order_interim.append(model_order_ext_var)
        model_coeff_full_interim.append(",".join(str(i) for i in coefficients_dict.items()))
        model_coeff_full_interim.append(",".join(str(i) for i in coefficients_dict_ext_var.items()))
        model_coeff_full ="//".join([str((items)) for items in model_coeff_full_interim])
        model_param_df = save_model_parameters(model_order_interim, ARIMA_Xreg, model_coeff_full)


    #            kick off pure time series module as no other variable would be used to forecast
    # TODO add pure time series models for timeframe where external variables are not available

    #        Runs if variables have orders as a relevant variable along with other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) == 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor
        forecast_output = forecast.forecast(model_arima_xreg, len(Xreg), xreg=Xreg.values)
        forecast_output_dict = dict(zip(forecast_output.names, list(forecast_output)))
        r_forecast_order = get_r_df(forecast_output_dict['mean'], None, None, arg.train_data.Date.max(),
                                    arg.parameters.tsfreq)
        df_forecast_order = forecast_df(arg.parameters, r_forecast_order['mean'], None, None, ARIMA_Xreg)

        #            modify data for pure time series models
        arg.train_data.loc[-1] = [arg.train_data["type"][0], arg.train_data["Part_number"][0],
                                  arg.train_data["Location"][0], r_forecast_order["mean"][0],
                                  r_forecast_order["Date"][0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)
        arg.hp = arg.hp - 1
        df_train_pts_int, df_forecast_pts, model_param_df = arima(arg)
        df_forecast = df_forecast_order.append(df_forecast_pts)
        model_coeff_full_interim.append(",".join(str(i) for i in coefficients_dict.items()))
        model_coeff_full ="//".join([str((items)) for items in model_coeff_full_interim])
        model_param_df[model_name] = ARIMA_Xreg
        model_param_df["arimax_coef"] = model_coeff_full
        model_order = (forecast_output[0])
        model_param_df = save_model_parameters(model_order, ARIMA_Xreg, model_coeff_full)

    #        runs with just external variables
    if (INT_VAR not in arg.var_names):
        Xreg = arg.test_var
        forecast_output = forecast.forecast(model_arima_xreg, len(Xreg), xreg=Xreg.values)
        forecast_output_dict = dict(zip(forecast_output.names, list(forecast_output)))
        r_forecast = get_r_df(forecast_output_dict['mean'], None, None, arg.train_data.Date.max(),
                              arg.parameters.tsfreq)
        r_forecast['upper'], r_forecast['lower'] = pi_function(arg.train_data[VAL_COL].values, df_train[forecast_value],
                                                   r_forecast['mean'], p= len(arg.train_var.columns))
        df_forecast_ext_var = forecast_df(arg.parameters, r_forecast['mean'], r_forecast['upper'], r_forecast['lower'] , ARIMA_Xreg)
        # pure time series if external variables are not available
        arg.hp = arg.hp-len(Xreg)
        if arg.hp != 0:
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": r_forecast['mean'],
                 "Date": df_forecast_ext_var.index})
            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts, model_param_df = arima(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
            model_coeff_full_interim.append(",".join(str(i) for i in coefficients_dict.items()))
            model_coeff_full ="//".join([str((items)) for items in model_coeff_full_interim])
            # model_param_df[model_name] = ARIMA_Xreg
            # model_param_df["arimax_coef"] = model_coeff_full
            model_order = (forecast_output[0])
            model_param_df = save_model_parameters(model_order, ARIMA_Xreg, model_coeff_full)
        else:
            df_forecast = df_forecast_ext_var
            model_order = (forecast_output[0])
            model_coeff_full_interim.append(",".join(str(i) for i in coefficients_dict.items()))
            model_coeff_full ="//".join([str((items)) for items in model_coeff_full_interim])
            model_param_df = save_model_parameters(model_order, ARIMA_Xreg, model_coeff_full)


    # TODO add pure time series models for timeframe where external variables are not available

    return df_train, df_forecast, model_param_df


def arima_xreg_training(arg):
    #    training module for Arima with external variables
    rdata = ts(arg.train_data[VAL_COL], frequency=arg.parameters.tsfreq)
    set_seed(1000)
    fit = forecast.auto_arima(rdata, xreg=arg.train_var.values)
    fit_dict = dict(zip(fit.names, list(fit)))
    fitted_values = fit_dict['fitted']
    coefficients = fit_dict['coef']
    coefficients_dict = dict(zip(coefficients.names, list(coefficients)))
    fitted = fitted_df(fitted_values, arg.train_data['Date'].min(), arg.parameters.tsfreq)
    mae = calculate_mae(fitted_values, arg.train_data[VAL_COL], ARIMA_Xreg)
    df_train = fit_df(fitted['fitted_values'], arg.train_data[VAL_COL], mae, ARIMA_Xreg)
    return fit, df_train, coefficients_dict


def arima(arg):
    #       ARIMA model without Regressors
    print("Running ARIMA. start_date: {}, last_date: {}, period: {}".format(arg.train_data.Date.iloc[0],
                                                                            arg.train_data.Date.iloc[-1], arg.hp))
    rdata = ts(arg.train_data[VAL_COL], frequency=arg.parameters.tsfreq, start=c(arg.train_data.Date.min().year, arg.train_data.Date.min().month) )
    set_seed(1000)
    fit = forecast.auto_arima(rdata)
    fit_dict = dict(zip(fit.names, list(fit)))
    fitted_values = fit_dict['fitted']
    fitted = fitted_df(fitted_values, arg.train_data['Date'].min(), arg.parameters.tsfreq)
    mae = calculate_mae(fitted['fitted_values'], arg.train_data[VAL_COL], ARIMA)
    df_train = fit_df(fitted['fitted_values'], arg.train_data[VAL_COL], mae, ARIMA)
    # forecasting block
    forecast_output = forecast.forecast(fit, arg.hp)
    forecast_dict = dict(zip(forecast_output.names, list(forecast_output)))
    r_df = get_r_df(forecast_dict['mean'], None, None, arg.train_data.Date.max(), arg.parameters.tsfreq)
    r_df['upper'], r_df['lower'] = pi_function(arg.train_data[VAL_COL].values, fitted['fitted_values'], r_df['mean'],
                                               p=0)
    df_forecast_pts = forecast_df(arg.parameters, r_df['mean'], r_df['upper'], r_df['lower'], ARIMA)
    model_order = (forecast_output[0])
    model_param_df = save_model_parameters(model_order, ARIMA, None)
    return df_train, df_forecast_pts, model_param_df
