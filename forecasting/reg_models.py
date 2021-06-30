from datetime import datetime
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from forecasting.utils import *
from forecasting.es_models import es_3

ts = robjects.r('ts')
pandas2ri.activate()
stats = importr('stats')
forecast = importr('forecast')
set_seed = robjects.r('set.seed')

VAL_COL = 'ship_qty'
INT_VAR = 'order_qty'
REG = 'REG'
PY_ES = 'ES'
PY_ES_2 = 'ES2'
PY_ES_3 = 'ES3'


def regression(arg):
    model_coeff_full = []
    model_order_full = []
    model_order_interim = []
    model_coeff_full_interim = []
    ##arg.metrics.add(key, datetime.now())
    print("Running REG. training from: {}, till: {}".format(arg.train_data.Date.iloc[0],
                                                            arg.train_data.Date.iloc[-1]))  # arg.hp
    #   train the model
    model_rg, df_train = regression_training(arg)

    #        Runs if variables have orders as a relevant variable along with other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) > 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor
        model_rg_output = model_rg.predict(Xreg)
        model_output_order = list(model_rg_output)

        arg.train_data.loc[-1] = [arg.train_data["type"][0], arg.train_data["Part_number"][0],
                                  arg.train_data["Location"][0], model_output_order[0], Xreg.index[0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)
        arg.train_var.loc[-1] = arg.test_var.iloc[0]
        arg.train_var.index = arg.train_data.index
        arg.train_var = arg.train_var.drop(columns=INT_VAR)  # c remove orders from the variables mix in train

        #            train model again without orders
        model_rg, df_train = regression_training(arg)
        Xreg = arg.test_var.iloc[1:, :]
        Xreg = Xreg.drop(columns=INT_VAR)
        model_rg_output = model_rg.predict(Xreg)
        model_output_ext_var = list(model_rg_output)
        model_output = model_output_order
        model_output.extend(model_output_ext_var)

        df_fcst = pd.DataFrame({"mean": model_output}, index=arg.test_var.index)
        upper, lower = pi_function(arg.train_data[VAL_COL], df_train[forecast_value],  df_fcst["mean"], p = len(arg.test_var.columns))
        df_forecast_ext_var = forecast_df(arg.parameters, df_fcst["mean"], upper, lower, REG)

        arg.hp = arg.hp - len(Xreg) - 1  # ()-1 for orders too)
        if arg.hp != 0:
            df_forecast_ext_var[forecast_value] = np.where(df_forecast_ext_var[forecast_value] < 0, 0,
                                                     df_forecast_ext_var[forecast_value])
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": model_output,
                 "Date":  arg.test_var.index}, index = arg.test_var.index)

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts, model_param_df = es_3(arg)
            dates_seq = pd.date_range(arg.train_data["Date"].max(), periods=arg.hp+1, freq="MS")
            dates_seq = dates_seq[1:]
            df_forecast_pts["dt"] = dates_seq
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
            model_coeff_full_interim.append(",".join(str(round(i, 2)) for i in model_rg.coef_.tolist()))
            model_coeff_full = "//".join([str((items)) for items in model_coeff_full_interim])
            model_param_df["reg_coef"] = model_coeff_full
            model_param_df["model_name"] = REG
        else:
            df_forecast = df_forecast_ext_var
            df_forecast = df_forecast_ext_var
            model_param_df = save_model_parameters(None, REG, model_rg.coef_)







    #        Runs if variables have only orders as a relevant variable and no other external variables
    if (INT_VAR in arg.var_names) and (len(arg.var_names) == 1):
        Xreg = arg.test_var.head(1)  # to take orders for the first month as a predictor
        model_rg_output = model_rg.predict(Xreg)
        model_output = model_rg_output

        df_fitted = pd.DataFrame({"mean": model_output}, index=Xreg.index)
        # upper, lower = pi_function(arg.val_data[VAL_COL], df_fitted, df_fitted["mean"], p = len(arg.val_var.columns))
        df_forecast_order = forecast_df(arg.parameters, df_fitted["mean"], None, None,
                                        REG)

        #                modify data for pure time series models
        arg.train_data.loc[-1] = [arg.train_data["type"][0],
                                  arg.train_data["Part_number"][0], arg.train_data["Location"][0], model_output[0],
                                  Xreg.index[0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)

        #                kick off pure time series module as no other variable would be used to forecast
        arg.hp = arg.hp - 1
        df_train_pts_int, df_forecast_pts, model_param_df = es_3(arg)
        df_forecast = df_forecast_order.append(df_forecast_pts)

        model_param_df["reg_coef"] = model_rg.coef_[0]
        model_param_df["model_name"] = REG



        #        runs with just external variables
    if INT_VAR not in arg.var_names:
        Xreg = arg.test_var
        model_rg_output = model_rg.predict(Xreg)
        model_output = model_rg_output
        df_fitted = pd.DataFrame({"mean": model_output}, index=Xreg.index)
        upper, lower = pi_function(arg.train_data[VAL_COL], df_fitted["mean"], df_fitted["mean"], p = len(arg.test_var.columns))
        df_forecast_ext_var = forecast_df(arg.parameters, df_fitted["mean"],   upper, lower, REG)

        arg.hp = arg.hp - len(Xreg) # ()-1 for orders too)
        if arg.hp != 0:
            df_forecast_ext_var[forecast_value] = np.where(df_forecast_ext_var[forecast_value] < 0, 0,
                                                     df_forecast_ext_var[forecast_value])
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": model_output,
                 "Date": Xreg.index}, index = Xreg.index)

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data = arg.train_data.set_index("Date", drop=False)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts_int, df_forecast_pts, model_param_df = es_3(arg)
            df_forecast = df_forecast_ext_var.append(df_forecast_pts)
            model_coeff_full_interim.append(",".join(str(round(i, 2)) for i in model_rg.coef_.tolist()))
            model_coeff_full = "//".join([str((items)) for items in model_coeff_full_interim])
            model_param_df["reg_coef"] = model_coeff_full
            model_param_df["model_name"] = REG
        else:
            df_forecast = df_forecast_ext_var
            model_param_df = save_model_parameters(None, REG, model_rg.coef_)


    return df_train, df_forecast, model_param_df


def regression_training(arg):
    #    Exponential Smoothening model with regressors

    regressor = LinearRegression()
    model_rg = regressor.fit(arg.train_var, arg.train_data[VAL_COL])
    # coef_df = pd.DataFrame({'var': arg.train_var.columns, 'coef': model_rg.coef_})
    model_reg_output = model_rg.predict(arg.train_var)
    df_fitted = pd.DataFrame({"mean": model_reg_output}, index=arg.train_data.index)

    mae = calculate_mae(df_fitted["mean"], arg.train_data[VAL_COL], REG)
    # upper, lower = pi_function(arg.train_data[VAL_COL], df_fitted, df_fitted["mean"], p = len(arg.train_var.columns))
    df_train = fit_df(df_fitted["mean"], arg.train_data[VAL_COL], mae, REG)

    return model_rg, df_train

