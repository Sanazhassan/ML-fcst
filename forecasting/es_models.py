from datetime import datetime
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from forecasting.utils import *

warnings.simplefilter('ignore', ConvergenceWarning)
ts = robjects.r('ts')
pandas2ri.activate()
stats = importr('stats')
forecast = importr('forecast')
set_seed = robjects.r('set.seed')

VAL_COL = 'ship_qty'
INT_VAR = 'order_qty'
ETS_Xreg = 'ETS_xreg'
ES2_Xreg = 'ES2_xreg'
ES1_Xreg = "ES1_Xreg"
PY_ES = 'ES'
PY_ES_2 = 'ES2'
PY_ES_3 = 'ES3'
model_coeff_full = []
model_order_full = []


# TODO fix frequency warning

# -----------------------------------------ES3 Xreg models----------------------------------#
# -----------------------------------------ES3 Xreg models----------------------------------#
# -----------------------------------------ES3 Xreg models----------------------------------#

def es3_xreg(arg):
    model_coeff_full = []
    model_order_full = []
    model_order_interim = []
    model_coeff_full_interim = []

    print("Running ETS_XREG. training from: {}, till: {} for {} periods".format(arg.train_data.Date.iloc[0],
                                                                                arg.train_data.Date.iloc[-1],
                                                                                arg.hp))  # arg.hp
    #   train the model
    model_hw, model_rg, df_train = es3_xreg_training(arg)

    #        Runs if variables have orders as a relevant variable along with other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) > 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor

        model_hw_output = model_hw.forecast(len(Xreg))
        model_rg_output = model_rg.predict(Xreg)
        model_output_order = model_rg_output + model_hw_output

        arg.train_data.loc[-1] = [arg.train_data["type"][0], arg.train_data["Part_number"][0],
                                  arg.train_data["Location"][0], model_output_order[0], model_output_order.index[0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)
        arg.train_var.loc[-1] = arg.test_var.iloc[0]
        arg.train_var.index = arg.train_data.index
        arg.train_var = arg.train_var.drop(columns=INT_VAR)  # c remove orders from the variables mix in train

        #            train model again without orders
        model_hw_ext_var, model_rg_ext_var, df_train = es3_xreg_training(arg)
        Xreg = arg.test_var.iloc[1:, :]
        Xreg = Xreg.drop(columns=INT_VAR)
        model_hw_output = model_hw_ext_var.forecast(len(Xreg))
        model_rg_output = model_rg_ext_var.predict(Xreg)
        model_output_ext_var = model_rg_output + model_hw_output
        model_output = model_output_order.append(model_output_ext_var)
        upper, lower = pi_function(arg.train_data[VAL_COL], df_train[forecast_value], model_output,
                                   p=len(arg.train_var.columns))
        df_forecast_ext_var = forecast_df(arg.parameters, model_output, upper, lower, ETS_Xreg)


        arg.hp  = arg.hp-len(Xreg)-1 # ()-1 for orders too)
        if arg.hp != 0:
            df_forecast_ext_var[forecast_value] = np.where(df_forecast_ext_var[forecast_value] <0, 0, df_forecast_ext_var[forecast_value] )
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": model_output_ext_var,
                 "Date": model_output_ext_var.index})

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts, model_param_df = es_3(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
        else:
            df_forecast = df_forecast_ext_var

        # output model parameters
        required_params = ["smoothing_level", "smoothing_trend"]
        model_order_order = [model_hw.params[p] for p in required_params]
        model_order_ext_var = [model_hw_ext_var.params[p] for p in required_params]
        model_order_interim.append(model_order_order)
        model_order_interim.append(model_order_ext_var)
        model_order_full = [",".join([str(round(items[0],2)) for items in model_order_interim]), ",".join([str(round(items[1],2)) for items in model_order_interim])]

        model_coeff_full_interim.append(",".join(str(round(i,2)) for i in model_rg.coef_.tolist()))
        model_coeff_full_interim.append(",".join(str(round(i,2)) for i in model_rg_ext_var.coef_.tolist()))
        model_coeff_full ="//".join([str((items)) for items in model_coeff_full_interim])
        model_param_df = save_model_parameters(model_order_full, ETS_Xreg, model_coeff_full)


    #        Runs if variables have only orders as a relevant variable and no other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) == 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor

        model_hw_output = model_hw.forecast(len(Xreg))
        model_rg_output = model_rg.predict(Xreg)
        model_output_order = np.round_(model_rg_output + model_hw_output)
        df_forecast_order = forecast_df(arg.parameters, model_output_order, None, None, ETS_Xreg)

        #                modify data for pure time series models
        arg.train_data.loc[-1] = [arg.train_data["type"][0],
                                  arg.train_data["Part_number"][0], arg.train_data["Location"][0],
                                  model_output_order[0], model_output_order.index[0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)

        #                kick off pure time series module as no other variable would be used to forecast
        arg.hp = arg.hp - 1
        df_train_pts_int, df_forecast_pts, model_param_df = es_3(arg)
        df_forecast = df_forecast_order.append(df_forecast_pts)

        # output model parameters
        model_param_df["reg_coef"] = model_rg.coef_[0]
        model_param_df["model_name"] = ETS_Xreg


    #        runs with just external variables
    if (INT_VAR not in arg.var_names):
        Xreg = arg.test_var
        model_hw_output = model_hw.forecast(len(Xreg))
        model_rg_output = model_rg.predict(Xreg)
        model_output_ext_var = np.round_(model_rg_output + model_hw_output)
        upper, lower = pi_function(arg.train_data[VAL_COL], df_train[forecast_value], model_output_ext_var,
                                   p=len(arg.train_var.columns))
        df_forecast_ext_var = forecast_df(arg.parameters, model_output_ext_var, upper, lower,
                                  ETS_Xreg)

        arg.hp  = arg.hp-len(Xreg)
        if arg.hp != 0:
            df_forecast_ext_var[forecast_value] = np.where(df_forecast_ext_var[forecast_value] <0, 0, df_forecast_ext_var[forecast_value] )
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": model_output_ext_var,
                 "Date": model_output_ext_var.index})

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts, model_param_df = es_3(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
        else:
            df_forecast = df_forecast_ext_var

        required_params = ["smoothing_level", "smoothing_trend"]
        model_order_ext_var = [model_hw.params[p] for p in required_params]
        model_order_interim.append(model_order_ext_var)
        model_order_full = [",".join([str(round(items[0],2)) for items in model_order_interim]), ",".join([str(round(items[1],2)) for items in model_order_interim])]
        model_coeff_full_interim.append(",".join(str(round(i,2)) for i in model_rg.coef_.tolist()))
        model_coeff_full ="|".join([str((items)) for items in model_coeff_full_interim])
        model_param_df = save_model_parameters(model_order_full, ETS_Xreg, model_coeff_full)

    return df_train, df_forecast, model_param_df


def es3_xreg_training(arg):
    #    Exponential Smoothening model with regressors

    regressor = LinearRegression()
    model_rg = regressor.fit(arg.train_var, arg.train_data[VAL_COL])
    coef_df = pd.DataFrame({'var': arg.train_var.columns, 'coef': model_rg.coef_})
    model_rg_fit = np.round_(model_rg.predict(arg.train_var))

    #        Residuals from regression for ets
    train_residual = arg.train_data[VAL_COL] - model_rg_fit
    #        Triple Exponential Smoothening model
    model_hw = ExponentialSmoothing(train_residual,
                                    trend='add',
                                    seasonal='add', seasonal_periods=12, freq='MS').fit()

    model_hw_fit = np.round_(model_hw.fittedvalues)
    df_fitted = model_rg_fit + model_hw_fit
    mae = calculate_mae(df_fitted, arg.train_data[VAL_COL].values, ETS_Xreg)
    df_train = fit_df(df_fitted, arg.train_data[VAL_COL], mae, ETS_Xreg)
    return model_hw, model_rg, df_train


# -----------------------------------------ES2 Xreg models----------------------------------#
# -----------------------------------------ES2 Xreg models----------------------------------#
# -----------------------------------------ES2 Xreg models----------------------------------#
def es2_xreg(arg):
    model_coeff_full = []
    model_order_full = []
    model_order_interim = []
    model_coeff_full_interim = []

    print("Running ES2_Xreg. training from: {}, till: {} for {} periods".format(arg.train_data.Date.iloc[0],
                                                                                arg.train_data.Date.iloc[-1],
                                                                                arg.hp))  # arg.hp
    #   train the model
    model_hw, model_rg, df_train = es2_xreg_training(arg)

    #        Runs if variables have orders as a relevant variable along with other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) > 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor

        model_hw_output = model_hw.forecast(len(Xreg))
        model_rg_output = model_rg.predict(Xreg)
        model_output_order = model_rg_output + model_hw_output

        arg.train_data.loc[-1] = [arg.train_data["type"][0], arg.train_data["Part_number"][0],
                                  arg.train_data["Location"][0], model_output_order[0], model_output_order.index[0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)
        arg.train_var.loc[-1] = arg.test_var.iloc[0]
        arg.train_var.index = arg.train_data.index
        arg.train_var = arg.train_var.drop(columns=INT_VAR)  # c remove orders from the variables mix in train

        #            train model again without orders
        model_hw_ext_var, model_rg_ext_var, df_train = es2_xreg_training(arg)
        Xreg = arg.test_var.iloc[1:, :]
        Xreg = Xreg.drop(columns=INT_VAR)
        model_hw_output = model_hw_ext_var.forecast(len(Xreg))
        model_rg_output = model_rg_ext_var.predict(Xreg)
        model_output_ext_var = model_rg_output + model_hw_output
        model_output = model_output_order.append(model_output_ext_var)
        upper, lower = pi_function(arg.train_data[VAL_COL], df_train[forecast_value], model_output,
                                   p=len(arg.train_var.columns))
        df_forecast_ext_var = forecast_df(arg.parameters, model_output, upper, lower, ES2_Xreg)


        arg.hp  = arg.hp-len(Xreg)-1 # ()-1 for orders too)
        if arg.hp != 0:
            df_forecast_ext_var[forecast_value] = np.where(df_forecast_ext_var[forecast_value] <0, 0, df_forecast_ext_var[forecast_value] )
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": model_output_ext_var,
                 "Date": model_output_ext_var.index})

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts, model_param_df = es_2(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
        else:
            df_forecast = df_forecast_ext_var

        # output model parameters
        required_params = ["smoothing_level", "smoothing_trend"]
        model_order_order = [model_hw.params[p] for p in required_params]
        model_order_ext_var = [model_hw_ext_var.params[p] for p in required_params]
        model_order_interim.append(model_order_order)
        model_order_interim.append(model_order_ext_var)
        model_order_full = [",".join([str(round(items[0],2)) for items in model_order_interim]), ",".join([str(round(items[1],2)) for items in model_order_interim])]

        model_coeff_full_interim.append(",".join(str(round(i,2)) for i in model_rg.coef_.tolist()))
        model_coeff_full_interim.append(",".join(str(round(i,2)) for i in model_rg_ext_var.coef_.tolist()))
        model_coeff_full ="//".join([str((items)) for items in model_coeff_full_interim])
        model_param_df = save_model_parameters(model_order_full, ES2_Xreg, model_coeff_full)


    #        Runs if variables have only orders as a relevant variable and no other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) == 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor

        model_hw_output = model_hw.forecast(len(Xreg))
        model_rg_output = model_rg.predict(Xreg)
        model_output_order = np.round_(model_rg_output + model_hw_output)
        df_forecast_order = forecast_df(arg.parameters, model_output_order, None, None, ES2_Xreg)

        #                modify data for pure time series models
        arg.train_data.loc[-1] = [arg.train_data["type"][0],
                                  arg.train_data["Part_number"][0], arg.train_data["Location"][0],
                                  model_output_order[0], model_output_order.index[0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)

        #                kick off pure time series module as no other variable would be used to forecast
        arg.hp = arg.hp - 1
        df_train_pts_int, df_forecast_pts, model_param_df = es_2(arg)
        df_forecast = df_forecast_order.append(df_forecast_pts)

        # output model parameters
        model_param_df["reg_coef"] = model_rg.coef_[0]
        model_param_df["model_name"] = ES2_Xreg


    #        runs with just external variables
    if (INT_VAR not in arg.var_names):
        Xreg = arg.test_var
        model_hw_output = model_hw.forecast(len(Xreg))
        model_rg_output = model_rg.predict(Xreg)
        model_output_ext_var = np.round_(model_rg_output + model_hw_output)
        upper, lower = pi_function(arg.train_data[VAL_COL], df_train[forecast_value], model_output_ext_var,
                                   p=len(arg.train_var.columns))
        df_forecast_ext_var = forecast_df(arg.parameters, model_output_ext_var, upper, lower,
                                  ES2_Xreg)

        arg.hp  = arg.hp-len(Xreg)
        if arg.hp != 0:
            df_forecast_ext_var[forecast_value] = np.where(df_forecast_ext_var[forecast_value] <0, 0, df_forecast_ext_var[forecast_value] )
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": model_output_ext_var,
                 "Date": model_output_ext_var.index})

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts, model_param_df = es_2(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
        else:
            df_forecast = df_forecast_ext_var

        required_params = ["smoothing_level", "smoothing_trend"]
        model_order_ext_var = [model_hw.params[p] for p in required_params]
        model_order_interim.append(model_order_ext_var)
        model_order_full = [",".join([str(round(items[0],2)) for items in model_order_interim]), ",".join([str(round(items[1],2)) for items in model_order_interim])]
        model_coeff_full_interim.append(",".join(str(round(i,2)) for i in model_rg.coef_.tolist()))
        model_coeff_full ="|".join([str((items)) for items in model_coeff_full_interim])
        model_param_df = save_model_parameters(model_order_full, ES2_Xreg, model_coeff_full)

    return df_train, df_forecast, model_param_df


def es2_xreg_training(arg):
    #    Exponential Smoothening model with regressors

    regressor = LinearRegression()
    model_rg = regressor.fit(arg.train_var, arg.train_data[VAL_COL])
    coef_df = pd.DataFrame({'var': arg.train_var.columns, 'coef': model_rg.coef_})
    model_rg_fit = np.round_(model_rg.predict(arg.train_var))
    #        Residuals from regression for ets
    train_residual = arg.train_data[VAL_COL] - model_rg_fit
    #        Triple Exponential Smoothening model
    model_hw = Holt(train_residual).fit()

    model_hw_fit = np.round_(model_hw.fittedvalues)
    df_fitted = model_rg_fit + model_hw_fit
    mae = calculate_mae(df_fitted, arg.train_data[VAL_COL].values, ES2_Xreg)
    df_train = fit_df(df_fitted, arg.train_data[VAL_COL], mae, ES2_Xreg)
    return model_hw, model_rg, df_train





# -----------------------------------------ES1 Xreg models----------------------------------#
# -----------------------------------------ES1 Xreg models----------------------------------#
# -----------------------------------------ES1 Xreg models----------------------------------#

def es1_xreg(arg):
    model_coeff_full = []
    model_order_full = []
    model_order_interim = []
    model_coeff_full_interim = []

    print("Running ES1_Xreg. training from: {}, till: {} for {} periods".format(arg.train_data.Date.iloc[0],
                                                                                arg.train_data.Date.iloc[-1],
                                                                                arg.hp))  # arg.hp
    #   train the model
    model_hw, model_rg, df_train = es1_xreg_training(arg)

    #        Runs if variables have orders as a relevant variable along with other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) > 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor

        model_hw_output = model_hw.forecast(len(Xreg))
        model_rg_output = model_rg.predict(Xreg)
        model_output_order = model_rg_output + model_hw_output

        arg.train_data.loc[-1] = [arg.train_data["type"][0], arg.train_data["Part_number"][0],
                                  arg.train_data["Location"][0], model_output_order[0], model_output_order.index[0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)
        arg.train_var.loc[-1] = arg.test_var.iloc[0]
        arg.train_var.index = arg.train_data.index
        arg.train_var = arg.train_var.drop(columns=INT_VAR)  # c remove orders from the variables mix in train

        #            train model again without orders
        model_hw_ext_var, model_rg_ext_var, df_train = es1_xreg_training(arg)
        Xreg = arg.test_var.iloc[1:, :]
        Xreg = Xreg.drop(columns=INT_VAR)
        model_hw_output = model_hw_ext_var.forecast(len(Xreg))
        model_rg_output = model_rg_ext_var.predict(Xreg)
        model_output_ext_var = model_rg_output + model_hw_output
        model_output = model_output_order.append(model_output_ext_var)
        upper, lower = pi_function(arg.train_data[VAL_COL], df_train[forecast_value], model_output,
                                   p=len(arg.train_var.columns))
        df_forecast_ext_var = forecast_df(arg.parameters, model_output, upper, lower, ES1_Xreg)


        arg.hp  = arg.hp-len(Xreg)-1 # ()-1 for orders too)
        if arg.hp != 0:
            df_forecast_ext_var[forecast_value] = np.where(df_forecast_ext_var[forecast_value] <0, 0, df_forecast_ext_var[forecast_value] )
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": model_output_ext_var,
                 "Date": model_output_ext_var.index})

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts, model_param_df = es(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
        else:
            df_forecast = df_forecast_ext_var

        # output model parameters
        required_params = ["smoothing_level", "smoothing_trend"]
        model_order_order = [model_hw.params[p] for p in required_params]
        model_order_ext_var = [model_hw_ext_var.params[p] for p in required_params]
        model_order_interim.append(model_order_order)
        model_order_interim.append(model_order_ext_var)
        model_order_full = [",".join([str(round(items[0],2)) for items in model_order_interim]), ",".join([str(round(items[1],2)) for items in model_order_interim])]

        model_coeff_full_interim.append(",".join(str(round(i,2)) for i in model_rg.coef_.tolist()))
        model_coeff_full_interim.append(",".join(str(round(i,2)) for i in model_rg_ext_var.coef_.tolist()))
        model_coeff_full ="//".join([str((items)) for items in model_coeff_full_interim])
        model_param_df = save_model_parameters(model_order_full, ES1_Xreg, model_coeff_full)


    #        Runs if variables have only orders as a relevant variable and no other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) == 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor

        model_hw_output = model_hw.forecast(len(Xreg))
        model_rg_output = model_rg.predict(Xreg)
        model_output_order = np.round_(model_rg_output + model_hw_output)
        df_forecast_order = forecast_df(arg.parameters, model_output_order, None, None, ES1_Xreg)

        #                modify data for pure time series models
        arg.train_data.loc[-1] = [arg.train_data["type"][0],
                                  arg.train_data["Part_number"][0], arg.train_data["Location"][0],
                                  model_output_order[0], model_output_order.index[0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)

        #                kick off pure time series module as no other variable would be used to forecast
        arg.hp = arg.hp - 1
        df_train_pts_int, df_forecast_pts, model_param_df = es(arg)
        df_forecast = df_forecast_order.append(df_forecast_pts)

        # output model parameters
        model_param_df["reg_coef"] = model_rg.coef_[0]
        model_param_df["model_name"] = ES1_Xreg


    #        runs with just external variables
    if (INT_VAR not in arg.var_names):
        Xreg = arg.test_var
        model_hw_output = model_hw.forecast(len(Xreg))
        model_rg_output = model_rg.predict(Xreg)
        model_output_ext_var = np.round_(model_rg_output + model_hw_output)
        upper, lower = pi_function(arg.train_data[VAL_COL], df_train[forecast_value], model_output_ext_var,
                                   p=len(arg.train_var.columns))
        df_forecast_ext_var = forecast_df(arg.parameters, model_output_ext_var, upper, lower,
                                  ES1_Xreg)

        arg.hp  = arg.hp-len(Xreg)
        if arg.hp != 0:
            df_forecast_ext_var[forecast_value] = np.where(df_forecast_ext_var[forecast_value] <0, 0, df_forecast_ext_var[forecast_value] )
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": model_output_ext_var,
                 "Date": model_output_ext_var.index})

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts, model_param_df = es(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
        else:
            df_forecast = df_forecast_ext_var

        required_params = ["smoothing_level", "smoothing_trend"]
        model_order_ext_var = [model_hw.params[p] for p in required_params]
        model_order_interim.append(model_order_ext_var)
        model_order_full = [",".join([str(round(items[0],2)) for items in model_order_interim]), ",".join([str(round(items[1],2)) for items in model_order_interim])]
        model_coeff_full_interim.append(",".join(str(round(i,2)) for i in model_rg.coef_.tolist()))
        model_coeff_full ="|".join([str((items)) for items in model_coeff_full_interim])
        model_param_df = save_model_parameters(model_order_full, ES1_Xreg, model_coeff_full)

    return df_train, df_forecast, model_param_df


def es1_xreg_training(arg):
    #    Exponential Smoothening model with regressors

    regressor = LinearRegression()
    model_rg = regressor.fit(arg.train_var, arg.train_data[VAL_COL])
    coef_df = pd.DataFrame({'var': arg.train_var.columns, 'coef': model_rg.coef_})
    model_rg_fit = np.round_(model_rg.predict(arg.train_var))

    #        Residuals from regression for ets
    train_residual = arg.train_data[VAL_COL] - model_rg_fit
    #        Triple Exponential Smoothening model
    model_hw = SimpleExpSmoothing(train_residual).fit()

    model_hw_fit = np.round_(model_hw.fittedvalues)
    df_fitted = model_rg_fit + model_hw_fit
    mae = calculate_mae(df_fitted, arg.train_data[VAL_COL].values, ES1_Xreg)
    df_train = fit_df(df_fitted, arg.train_data[VAL_COL], mae, ES1_Xreg)
    return model_hw, model_rg, df_train






# ---------------------------------------------------------------Time series models----------------------------------#
# ---------------------------------------------------------------Time series models----------------------------------#
# ---------------------------------------------------------------Time series models----------------------------------#



def es(arg):
    print("Running ESM1 model. start_date: {}, last_date: {}".format(arg.train_data.Date.iloc[0],
                                                                 arg.train_data.Date.iloc[-1]))

    model = SimpleExpSmoothing(arg.train_data[VAL_COL]).fit()
    df_fitted = np.round_(model.fittedvalues)
    mae = calculate_mae(df_fitted, arg.train_data[VAL_COL], PY_ES)
    df_train = fit_df(df_fitted, arg.train_data[VAL_COL], mae, PY_ES)
    model_output = np.round_(model.forecast(arg.hp))
    upper, lower = pi_function(arg.train_data[VAL_COL], df_fitted, model_output, p=0)
    df_forecast = forecast_df(arg.parameters, model_output, upper, lower, PY_ES)
    # output model parameters
    required_params = ["smoothing_level", "smoothing_trend"]
    model_order = [round(model.params[p],2) for p in required_params]
    model_param_df = save_model_parameters(model_order, PY_ES, None)
    return df_train, df_forecast, model_param_df


def es_2(arg):
    # Add key from the metric file
    ##arg.metrics.add(key, datetime.now())
    print("Running ESM2 model. start_date: {}, last_date: {}".format(arg.train_data.Date.iloc[0],
                                                                     arg.train_data.Date.iloc[-1]))

    # arg.train_data.index = pd.DatetimeIndex(arg.train_data.index).to_period('MS')
    model = Holt(arg.train_data[VAL_COL]).fit()
    df_fitted = np.round_(model.fittedvalues)
    mae = calculate_mae(df_fitted, arg.train_data[VAL_COL], PY_ES_2)
    df_train = fit_df(df_fitted, arg.train_data[VAL_COL], mae, PY_ES_2)
    model_output = np.round_(model.forecast(arg.hp))
    upper, lower = pi_function(arg.train_data[VAL_COL], df_fitted, model_output, p=0)
    df_forecast = forecast_df(arg.parameters, model_output, upper, lower, PY_ES_2)
    # output model parameters
    required_params = ["smoothing_level", "smoothing_trend"]
    model_order = [round(model.params[p],2) for p in required_params]
    model_param_df = save_model_parameters(model_order, PY_ES_2, None)
    return df_train, df_forecast, model_param_df


def es_3(arg):
    # Add key from the metric file
    ##arg.metrics.add(key, datetime.now())
    print("Running ESM3 model. start_date: {}, last_date: {}".format(arg.train_data.Date.iloc[0],
                                                                     arg.train_data.Date.iloc[-1]))
    try:
        model = ExponentialSmoothing(arg.train_data[VAL_COL], trend='add', seasonal='add', seasonal_periods=12).fit()
    except:
        model = ExponentialSmoothing(arg.train_data[VAL_COL], trend='add', seasonal_periods=12).fit()

    df_fitted = np.round_(model.fittedvalues)
    mae = calculate_mae(df_fitted, arg.train_data[VAL_COL], PY_ES_3)
    df_train = fit_df(df_fitted, arg.train_data[VAL_COL], mae, PY_ES_3)
    model_output = np.round_(model.forecast(arg.hp))
    upper, lower = pi_function(arg.train_data[VAL_COL], df_fitted, model_output, p=0)
    df_forecast = forecast_df(arg.parameters, model_output, upper, lower, PY_ES_3)
        # output model parameters
    required_params = ["smoothing_level", "smoothing_trend"]
    model_order = [round(model.params[p],2) for p in required_params]
    model_param_df = save_model_parameters(model_order, PY_ES_3, None)
    return df_train, df_forecast, model_param_df
