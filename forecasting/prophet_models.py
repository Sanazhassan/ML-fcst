# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:46:21 2021

@author: Sana Hassan
"""

from fbprophet import Prophet
from forecasting.utils import *
from dateutil.relativedelta import relativedelta

VAL_COL = 'ship_qty'
INT_VAR = 'order_qty'
PROPHET_Xreg = 'prophet_xreg'
PROPHET = 'prophet'


def prophet_xreg(arg):
    print("Running PROPHET - XREG. start_date: {}, last_date: {}".format(arg.train_data.Date.iloc[0],
                                                                         arg.train_data.Date.iloc[-1]))
    df_train, m = prophet_xreg_training(arg)

    if INT_VAR not in arg.var_names:
        # with  external variables
        Xreg = arg.test_var
        future_range = pd.date_range(arg.test_var.index.min(), periods=len(Xreg), freq='MS')
        future = pd.DataFrame({'ds': future_range})
        for v in arg.var_names:
            data = Xreg[v].values
            future[v] = data
        fcst_ext_var = m.predict(future)
        fcst_ext_var = fcst_ext_var.set_index('ds', drop=False)
        fcst_ext_var = fcst_ext_var.last(str(len(Xreg)) + 'M')

        # only pure time series
        arg.hp = arg.hp - len(Xreg)
        if arg.hp != 0:
            fcst_start_date = Xreg.index.max() + relativedelta(months=1)
            fcst_ext_var["yhat"] = np.where(fcst_ext_var["yhat"] < 0, 0, fcst_ext_var["yhat"])
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": fcst_ext_var["yhat"],
                 "Date": fcst_ext_var["yhat"].index})

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts, m_pts = prophet(arg)

            future_range = pd.date_range(fcst_start_date, periods=arg.hp, freq='MS')
            future = pd.DataFrame({'ds': future_range})
            fcst_pts = m_pts.predict(future)
            fcst_pts = fcst_pts.set_index('ds', drop=False)
            fcst_pts = fcst_pts.last(str(arg.hp) + 'M')
            fcst_full = fcst_ext_var.append(fcst_pts)
        else:
            fcst_full = fcst_ext_var

        fcst_full['yhat'] = np.where(fcst_full['yhat'] < 0, 0, round(fcst_full['yhat'], 0))
        fcst_full['yhat_upper'] = np.where(fcst_full['yhat_upper'] < 0, 0, round(fcst_full['yhat_upper'], 0))
        fcst_full['yhat_lower'] = np.where(fcst_full['yhat_lower'] < 0, 0, round(fcst_full['yhat_lower'], 0))


    if (INT_VAR in arg.var_names) and (len(arg.var_names) > 1):

        future_range = pd.date_range(arg.test_var.index.min(), periods=1, freq='MS')
        future = pd.DataFrame({'ds': future_range})
        # with orders and external variables
        Xreg = arg.test_var.head(1)
        for v in arg.var_names:
            data = Xreg[v].values
            future[v] = data
        fcst_order = m.predict(future)
        fcst_order = fcst_order.set_index('ds', drop=False)
        fcst_order = fcst_order.last(str(1) + 'M')

        # with only external variables
        arg.train_data.loc[-1] = [arg.train_data["type"][0],
                                  arg.train_data["Part_number"][0], arg.train_data["Location"][0],
                                  fcst_order['yhat'].values[0],
                                  fcst_order['yhat'].index[0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)
        arg.train_var = arg.train_var.append(arg.test_var.head(1))
        arg.train_var = arg.train_var.drop(columns=INT_VAR)
        arg.var_names.remove(INT_VAR)
        df_train_ext_var, m_ext_var = prophet_xreg_training(arg)
        Xreg = arg.test_var.tail(-1)
        Xreg = Xreg.drop(columns=INT_VAR)
        future_range = pd.date_range(Xreg.index.min(), periods=len(Xreg), freq='MS')
        future = pd.DataFrame({'ds': future_range})
        for v in arg.var_names:
            data = Xreg[v].values
            future[v] = data
        fcst_ext_var = m_ext_var.predict(future)
        fcst_ext_var = fcst_ext_var.set_index('ds', drop=False)
        fcst_ext_var = fcst_ext_var.last(str(len(Xreg)) + 'M')
        fcst_ext_var = fcst_order.append(fcst_ext_var)

        # only pure time series
        arg.hp = arg.hp - len(Xreg) - 1
        if arg.hp != 0:
            fcst_start_date = Xreg.index.max() + relativedelta(months=1)
            fcst_ext_var["yhat"] = np.where(fcst_ext_var["yhat"] < 0, 0, fcst_ext_var["yhat"])
            forecast_to_append = pd.DataFrame(
                {"type": arg.train_data["type"][0], "Part_number": arg.train_data["Part_number"][0],
                 "Location": arg.train_data["Location"][0], "ship_qty": fcst_ext_var["yhat"],
                 "Date": fcst_ext_var["yhat"].index})

            arg.train_data = arg.train_data.append(forecast_to_append)
            arg.train_data["Date"] = arg.train_data.index
            df_train_pts, m_pts = prophet(arg)

            future_range = pd.date_range(fcst_start_date, periods=arg.hp, freq='MS')
            future = pd.DataFrame({'ds': future_range})
            fcst_pts = m_pts.predict(future)
            fcst_pts = fcst_pts.set_index('ds', drop=False)
            fcst_pts = fcst_pts.last(str(arg.hp) + 'M')
            fcst_full = fcst_ext_var.append(fcst_pts)
        else:
            fcst_full = fcst_ext_var

        fcst_full['yhat'] = np.where(fcst_full['yhat'] < 0, 0, round(fcst_full['yhat'], 0))
        fcst_full['yhat_upper'] = np.where(fcst_full['yhat_upper'] < 0, 0, round(fcst_full['yhat_upper'], 0))
        fcst_full['yhat_lower'] = np.where(fcst_full['yhat_lower'] < 0, 0, round(fcst_full['yhat_lower'], 0))



    if (INT_VAR in arg.var_names) and (len(arg.var_names) == 1):
        future_range = pd.date_range(arg.test_var.index.min(), periods=1, freq='MS')
        future = pd.DataFrame({'ds': future_range})
        # with orders and external variables
        Xreg = arg.test_var.head(1)
        for v in arg.var_names:
            data = Xreg[v].values
            future[v] = data
        fcst_order = m.predict(future)
        fcst_order = fcst_order.set_index('ds', drop=False)
        fcst_order = fcst_order.last(str(1) + 'M')

        # with only external variables
        arg.train_data.loc[-1] = [arg.train_data["type"][0],
                                  arg.train_data["Part_number"][0], arg.train_data["Location"][0],
                                  fcst_order['yhat'].values[0],
                                  fcst_order['yhat'].index[0]]
        arg.train_data = arg.train_data.set_index("Date", drop=False)

        df_train_pts,  m_pts = prophet(arg)
        arg.hp = arg.hp - 1
        fcst_start_date = Xreg.index.max() + relativedelta(months=1)
        future_range = pd.date_range(fcst_start_date, periods=(arg.hp), freq='MS')
        future = pd.DataFrame({'ds': future_range})
        fcst_pts = m_pts.predict(future)
        fcst_pts = fcst_pts.set_index('ds', drop=False)
        fcst_pts = fcst_pts.last(str(arg.hp) + 'M')
        fcst_full = fcst_order.append(fcst_pts)

        fcst_full['yhat'] = np.where(fcst_full['yhat'] < 0, 0, round(fcst_full['yhat'], 0))
        fcst_full['yhat_upper'] = np.where(fcst_full['yhat_upper'] < 0, 0, round(fcst_full['yhat_upper'], 0))
        fcst_full['yhat_lower'] = np.where(fcst_full['yhat_lower'] < 0, 0, round(fcst_full['yhat_lower'], 0))


    df_forecast = forecast_df(arg.parameters, fcst_full['yhat'], fcst_full['yhat_upper'], fcst_full['yhat_lower'],
                              PROPHET_Xreg)

    return df_train, df_forecast







def prophet_xreg_training(arg):
    arg.train_data_interim = pd.concat([arg.train_data[VAL_COL], arg.train_var], axis=1).dropna(axis=0).reset_index(
        drop=False)
    arg.train_data_interim = arg.train_data_interim.rename(columns={'Date': 'ds', 'ship_qty': 'y'})
    arg.train_data_interim = arg.train_data_interim.set_index('ds', drop=False)

    model = Prophet(growth='linear',
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.01,  # changepoint_prior_scale=0.01,
                seasonality_prior_scale=15,  # seasonality_prior_scale= 20,
                seasonality_mode='multiplicative')  # seasonality_mode='additive')

    for v in arg.var_names:
        model.add_regressor(v)

    print(model)
    model_fit = model.fit(arg.train_data_interim)
    df_fitted = model_fit.predict(arg.train_data_interim)
    df_fitted = df_fitted.set_index('ds', drop=False)
    mae = calculate_mae(df_fitted['yhat'].values, arg.train_data_interim['y'], PROPHET_Xreg)
    df_train = fit_df(df_fitted['yhat'], arg.train_data[VAL_COL], mae, PROPHET_Xreg)
    return df_train, model







def prophet(arg):
    arg.train_data_interim = arg.train_data.rename(columns={'Date': 'ds', 'ship_qty': 'y'})
    arg.train_data_interim = arg.train_data_interim.set_index('ds', drop=False)

    model = Prophet(growth='linear',
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.01,  # changepoint_prior_scale=0.01,
                seasonality_prior_scale=15,  # seasonality_prior_scale= 20,
                seasonality_mode='multiplicative')  # seasonality_mode='additive')

    print(model)
    m_fit = model.fit(arg.train_data_interim)
    df_fitted = m_fit.predict(arg.train_data_interim)
    df_fitted = df_fitted.set_index('ds', drop=False)
    df_fitted = df_fitted.last(str(len(arg.train_data_interim)) + 'M')
    mae = calculate_mae(df_fitted['yhat'].values, arg.train_data_interim['y'], PROPHET)
    df_train = fit_df(df_fitted['yhat'], arg.train_data[VAL_COL], mae, PROPHET)
    return df_train,  model
