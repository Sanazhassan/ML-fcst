# -*- coding: utf-8 -*-
"""
Created on Mon Feb  10 11:20:17 2020

@author: Sana Hassan
"""

from os import getcwd
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from rpy2.robjects import pandas2ri
import math
from data_manager.table_names import *
pd.set_option("mode.chained_assignment", None)
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from database.sql import save_to_db


def get_r_df(mean, lower, upper, max_date, freq):
    """
    Changes R object to Pandas dataFrame
    """

    if freq == 12:
        index = pd.date_range(start=max_date, periods=len(mean) + 1, freq='MS')[1:]
    elif freq == 52:
        index = pd.date_range(start=max_date, periods=len(mean) + 1, freq='W')[1:]

    mean = pandas2ri.ri2py(mean)
    # mean = ro.conversion.rpy2py(mean)

    if lower is not None:
        lower = pandas2ri.ri2py(lower)
        # lower = pandas2ri.rpy2py(lower)
        lower_df = pd.DataFrame(lower, columns=['0', '1'])
        lower_df = lower_df['1']  # 0:80%,1: 95%
    else:
        lower_df = None

    if upper is not None:
        upper = pandas2ri.ri2py(upper)
        # upper = pandas2ri.rpy2py(upper)
        upper_df = pd.DataFrame(upper, columns=['0', '1'])
        upper_df = upper_df['1']  # for 95%
    else:
        upper_df = None

    r_df = pd.DataFrame({'Date': index, 'mean': mean, 'upper': upper_df, 'lower': lower_df})
    r_df = r_df.set_index('Date', drop=False)
    return r_df




def get_pydf(forecast_output, max_date):
    index = pd.date_range(start=max_date, periods=len(forecast_output) + 1, freq='MS')[1:]
    py_df = pd.Series(forecast_output, index=index)
    return py_df


def fitted_df(fitted_data, min_date, freq):
    df = pd.DataFrame({'dt': pd.date_range(start=(min_date - relativedelta(months=1)),
                                           periods=len(fitted_data) + 1,
                                           freq='MS')[1:], 'fitted_values': fitted_data})
    df = df.set_index('dt', drop=False)
    return df


def add_metadata(location, part_number, type_skus, period, var_used, line_item, training_time, start_time, forecast_id, *dfs):
    """
    Adds Location, part_number , period and time of run  the dataframe
    """
    for df in dfs:
        df[ship_loc] = location
        df[part_num] = part_number
        df["Type"] = type_skus
        df[forecast_horizon] = period
        if var_used is not None:
            df["Var_used"] = ",".join(var_used)
        else:
            df["Var_used"] = None
        df[division_id] = line_item
        df[forecast_run_date] = start_time
        df[training_run_date] = training_time
        df[fcast_id] = forecast_id
        df[record_id] = str(line_item) + "_" + str(part_number) + "_" + str(location)



def calculate_mae(pred, true, name):
    """
    Calculates Mean Absolute Percentage Error from predicted and true values
    """
    evaluation_df = pd.DataFrame({"true": true, "pred": pred})
    evaluation_df = evaluation_df.dropna()
    mae = round(abs(pred - true), 2)
    # av_mae = mean_absolute_error(evaluation_df["true"], evaluation_df["pred"])
    return mae.values


def forecast_df(parameters, pred_mean, pred_upper, pred_lower, name):
    """
    Creates a common dataframe for all the forecasts coming from different model for
    the ease of final storage in the DB
    """

    f_df = pd.DataFrame({forecast_date: pred_mean.index, model_name: name, forecast_value: pred_mean.values, upper_ci: pred_upper,
                         lower_ci: pred_lower, actual_value: None}).reset_index(drop=True)
    f_df = f_df.astype(object).where(pd.notnull(f_df), None)
    return f_df


def fit_df(pred, true, mae, name):
    """
    Creates a common dataframe for all the fitted values coming from different models for
    the ease of final storage in the DB
    """
    f_df = pd.DataFrame(
        {'dt': pred.index, 'model_name': name, forecast_value: np.round_(pred.values), "actuals": true}).reset_index(
        drop=True)
    f_df = f_df.astype(object).where(pd.notnull(f_df), None)
    return f_df





def create_lags(data, variables, correlated_var):
    """
    Creates dataframe for all the lags defined in the combination list file,
    respective to the location part_number
    """

    full_name = []
    var_full = pd.DataFrame()

    #    create Date column in variable
    variables = variables.reset_index()
    variables = variables.sort_values(by='Date', ascending=True)

    for index, rows in correlated_var.iterrows():
        name = rows['Selected_variables']
        lag = int(rows['Lag'])
        var = variables[['Date', name]]
        lagged = var[name]
        lagged = lagged.shift(lag)
        full_name.append(name)
        var_full = pd.concat([var_full, lagged], axis=1)

    var_data = pd.DataFrame(var_full)
    var_data.columns = full_name
    var_data['Date'] = variables['Date']
    var_data = var_data[(var_data["Date"] >= data.index.min())]
    var_data = var_data.apply(lambda x: x.fillna(x.mean()), axis=0)
    var_data = var_data.sort_values(by='Date', ascending=True)
    var_data = var_data.set_index('Date', drop=False)
    return var_data, full_name


def pi_function(train_data, model_fit, model_output, p):
    # number of xregs
    t = 1.99  # Two tail t distribution value at a = 0.025, and df = n-p-1
    sum_errs = np.sum((train_data - model_fit) ** 2)
    # stdev = np.sqrt(1 / (len(train_data) - 3) * sum_errs)
    # stdev = sqrt(1/(len(train_data)-p-1) * sum_errs)
    se_model = np.sqrt(sum_errs / (len(train_data) - p - 1))
    se_mean = se_model / (np.sqrt(len(train_data)))
    se_fcst = np.sqrt(se_model ** 2 + se_mean ** 2)
    model_output_upper_pi = np.round_(model_output + se_fcst * t)
    model_output_lower_pi = np.round_(model_output - se_fcst * t)
    return model_output_upper_pi.values, model_output_lower_pi.values


def clean_dataframe(df_fit, df_fcast):
    df_fit[forecast_value] = np.round_((df_fit[forecast_value].astype(float)), 0)
    df_fit[forecast_value] = np.where(df_fit[forecast_value] < 0, 0, df_fit[forecast_value])
    df_fcast[forecast_value] = np.round_((df_fcast[forecast_value].astype(float)), 0)
    df_fcast[forecast_value] = np.where(df_fcast[forecast_value] < 0, 0, df_fcast[forecast_value])
    return df_fit, df_fcast


def save_error_skus(pn, wn, best_model, training_time, start_time, line_item, forecast_id, error_details, name):
    print('saving errored sku part_number {} at location {}'.format(pn, wn))
    error_sku = {'Part_number': pn, 'Shipping_Location': wn,
                 "Best_model": best_model,
                 training_run_date: training_time,
                 forecast_run_date: start_time,
                 division_id: line_item,
                 fcast_id: forecast_id, 'Description': error_details}
    df_error = pd.DataFrame.from_dict([error_sku])
    save_to_db(df_error, kind= name)





def index_marks(nrows, chunk_size):
    return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)


def split_dataframe(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)



def save_model_parameters(model_param_raw, model_name, coefficients= None):
    model_param_df_cols = ["S_pqd", "pqd", "alpha","beta", "arimax_coef", "reg_coef" ]
    param_data = pd.DataFrame(columns=model_param_df_cols)

    if model_name == ARIMA_Xreg or model_name == ARIMA:
        model_param = [i for i in str(model_param_raw[0]).split(" ") if i.startswith('ARIMA')]

        if len(model_param[0]) == 23:
            model_order = model_param[0][5:][:-4]
            param_value = pd.Series([model_order[7:], model_order[:7], None, None, coefficients, None], model_param_df_cols)
            param_data = param_data.append(param_value, ignore_index=True)
        else:
            param_value = pd.Series([None, model_param[0][5:], None, None, coefficients, None], model_param_df_cols)
            param_data = param_data.append(param_value, ignore_index=True)

    if model_name == PY_ES or model_name == PY_ES_2 or model_name == PY_ES_3 or model_name == ETS_Xreg or model_name == ES1_Xreg or model_name == ES2_Xreg:

        param_value = pd.Series([None, None, model_param_raw[0],  model_param_raw[1], None, coefficients], model_param_df_cols)
        param_value = param_value.where(pd.notnull(param_value), None)
        param_data = param_data.append(param_value, ignore_index=True)

    if model_name == REG:
        param_value = pd.Series([None, None, None, None, None, coefficients],
                                model_param_df_cols)
        param_value = param_value.where(pd.notnull(param_value), None)
        param_data = param_data.append(param_value, ignore_index=True)

    param_data["model_name"] = model_name
    param_data  =  param_data.where(pd.notnull(param_data), None)
    return param_data
