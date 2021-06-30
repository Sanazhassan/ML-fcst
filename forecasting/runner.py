# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:17:51 2020

@author: Sana Hassan
"""

import numpy as np
import copy
import pandas as pd
from database.sql import save_to_db
from forecasting.utils import *
from forecasting.nn_models import nn_xreg, nn, NN, NN_Xreg
from forecasting.arima_models import arima_xreg, arima, ARIMA, ARIMA_Xreg
from forecasting.es_models import es3_xreg, es_3, es_2, es, ETS_Xreg, PY_ES, PY_ES_2, PY_ES_3, ES1_Xreg, ES2_Xreg, \
    es1_xreg, es2_xreg
from forecasting.reg_models import regression, REG
from forecasting.crostontsb import croston_function, tsb_function, CROSTON, TSB
from forecasting.prophet_models import PROPHET, PROPHET_Xreg, prophet_xreg, prophet
from forecasting.input import ModelInput
from data_manager.data_split import data_split
from forecasting.ma_model import moving_average_function, MA
from data_manager.table_names import *


class Runner:
    # TODO: Split the class into two - Class(Models), Class(run)

    """
    A class object that helps run through all the steps involved during a forecasting process ranging from data
     pre-processing to model selection and generating final forecast output from the best model.
    """

    def __init__(self, best_model, selected_variables, dataframe, var_subset, var_names, parameters, wn, pn, metrics,
                 start_time):
        """
        A class object that helps run through all the steps involved during a forecasting process ranging from data
        pre-processing to model selection and generating final forecast output from the best model.
        """

        # if var_subset is None:
        #     data_usable = dataframe
        # else:
        #     var_subset = var_subset.dropna()
        #     data_usable = dataframe[dataframe.index >= var_subset['Date'].min()] # to match with lagged variables
        self.df = dataframe
        self.variables = var_subset
        self.start_date = parameters.start_date_filter
        self.holdout_periods = parameters.holdout_periods
        self.fcast_periods = parameters.fcast_periods
        self.line_item = parameters.division_filter[0]
        self.var_names = var_names
        self.location = wn
        self.part_number = pn
        self.metrics = metrics
        self.parameters = parameters
        self.training_time_of_run = parameters.training_time_of_run
        self.start_time = start_time
        self.forecast_id = parameters.forecast_id
        self.best_model = best_model[["Best_model"]].values[0][0]
        self.best_model_eval_metric = best_model[["eval_metric"]].values[0][0]
        self.type_sku = self.df[["type"]].values[0][0]
        self.selected_variables = selected_variables

    def run(self):
        """
        A class object that helps run through all the steps involved during a forecasting process ranging from data preprocessing
        to model selection and generating final forecast output from the best model.
        """

        self.run_blind_forecast()

    # *****************************************************************************************************************#
    #                                                Methods Start here                                                #
    # *****************************************************************************************************************#

    # **************************************************** TRAINING ****************************************************#
    def run_blind_forecast(self):
        # Re-train the data with the selected model to give forecast for the desired time period
        df_forecast = pd.DataFrame()
        df_final_fit = pd.DataFrame()
        model_param_final = pd.DataFrame()

        # Run respective models with best selected model for the entire period from 2011-2018 and forecast in 2019
        restart = True
        while restart:
            for f_period in self.fcast_periods:

                # Inactive SKUs
                # if self.best_model == inactive:
                #
                #     df_fcast = forecast_df(self.parameters, 0 * f_period, 0 * f_period, 0 * f_period, inactive)
                #     add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                #                  self.line_item, self.training_time_of_run, self.start_time, self.forecast_id, df_fcast)
                #     df_forecast = df_forecast.append(df_fcast)
                #     if f_period == min(self.fcast_periods):
                #         restart = False

                if (self.best_model_eval_metric == 'short_time_series') or (self.best_model == inactive):
                    df_forecast = use_moving_average_for_exception(self, f_period)
                    if f_period == min(self.fcast_periods):
                        restart = False

                else:
                    try:
                        train_data, train_var, test_var = data_split(self.df, self.variables, self.var_names, f_period)

                        # checks if variables chosen are correect
                        if (self.var_names is not None) and (test_var is not None):
                            if set(self.var_names) == set(test_var.columns.to_list()):
                                self.var_names = self.var_names
                            else:
                                self.var_names = test_var.columns.to_list()
                        elif (self.var_names is not None) and (test_var is None):
                            self.var_names = None

                        if (self.var_names is not None) and (test_var is not None):
                            model_input = ModelInput(self.location, self.part_number, self.parameters, self.var_names,
                                                     train_data, train_var, f_period, test_var)

                            if self.best_model == ARIMA_Xreg:
                                try:
                                    df_fit, df_fcast, model_param_df = arima_xreg(copy.deepcopy(model_input))
                                except:
                                    save_error_skus(self.part_number, self.location, self.best_model, self.training_time_of_run,
                                                    self.start_time, self.line_item, self.forecast_id, "Arimax model couldnt converge using ETS_Xreg",
                                                    name=errored_sku_table)
                                    df_fit, df_fcast, model_param_df = es3_xreg(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item, self.training_time_of_run, self.start_time, self.forecast_id, df_fit, df_fcast,
                                             model_param_df)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                model_param_final = model_param_final.append(model_param_df)
                                if f_period == min(self.fcast_periods):
                                    restart = False

                            elif self.best_model == NN_Xreg:
                                np.random.seed(1000)
                                df_fit, df_fcast = nn_xreg(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id, df_fit, df_fcast)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                if f_period == min(self.fcast_periods):
                                    restart = False

                            elif self.best_model == ETS_Xreg:
                                np.random.seed(123)
                                df_fit, df_fcast, model_param_df = es3_xreg(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id,  df_fit, df_fcast, model_param_df)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                model_param_final = model_param_final.append(model_param_df)
                                if f_period == min(self.fcast_periods):
                                    restart = False

                            elif self.best_model == ES1_Xreg:
                                np.random.seed(123)
                                df_fit, df_fcast, model_param_df = es1_xreg(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item, self.training_time_of_run, self.start_time, self.forecast_id, df_fit, df_fcast, model_param_df)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                model_param_final = model_param_final.append(model_param_df)
                                if f_period == min(self.fcast_periods):
                                    restart = False

                            elif self.best_model == ES2_Xreg:
                                np.random.seed(123)
                                df_fit, df_fcast, model_param_df = es2_xreg(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period,
                                             self.var_names, self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id, df_fit, df_fcast, model_param_df)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                model_param_final = model_param_final.append(model_param_df)
                                if f_period == min(self.fcast_periods):
                                    restart = False

                            elif self.best_model == PROPHET_Xreg:
                                np.random.seed(123)
                                df_fit, df_fcast = prophet_xreg(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period,
                                             self.var_names, self.line_item,
                                             self.training_time_of_run, self.start_time,self.forecast_id, df_fit, df_fcast)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                if f_period == min(self.fcast_periods):
                                    restart = False


                            elif self.best_model == REG:
                                df_fit, df_fcast, model_param_df = regression(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id, df_fit, df_fcast, model_param_df)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                model_param_final = model_param_final.append(model_param_df)
                                if f_period == min(self.fcast_periods):
                                    restart = False


                            else:
                                save_error_skus(self.part_number, self.location, self.best_model, self.training_time_of_run,
                                                self.start_time, self.line_item, self.forecast_id,
                                                "didnot find a Xreg-model to run, using moving average",
                                                name=errored_sku_table)
                                df_forecast = use_moving_average_for_exception(f_period)
                                if f_period == min(self.fcast_periods):
                                    restart = False

                        else:

                            model_input = ModelInput(self.location, self.part_number, self.parameters, self.var_names,
                                                     train_data, None, f_period, None)

                            if self.best_model == PY_ES:
                                df_fit, df_fcast, model_param_df = es(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id, df_fit, df_fcast, model_param_df)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                model_param_final = model_param_final.append(model_param_df)
                                if f_period == min(self.fcast_periods):
                                    restart = False
                            #
                            #
                            elif self.best_model == PY_ES_2:
                                df_fit, df_fcast, model_param_df = es_2(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id, df_fit, df_fcast, model_param_df)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                model_param_final = model_param_final.append(model_param_df)
                                if f_period == min(self.fcast_periods):
                                    restart = False


                            elif (self.best_model == PY_ES_3) or (self.best_model == ETS_Xreg) or (
                                    self.best_model == REG):
                                df_fit, df_fcast, model_param_df = es_3(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time,self.forecast_id, df_fit, df_fcast, model_param_df)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                model_param_final = model_param_final.append(model_param_df)
                                if f_period == min(self.fcast_periods):
                                    restart = False

                            #
                            #
                            elif self.best_model == ARIMA or (self.best_model == ARIMA_Xreg):
                                try:
                                    df_fit, df_fcast, model_param_df = arima(copy.deepcopy(model_input))
                                except:
                                    save_error_skus(self.part_number, self.location, self.best_model, self.training_time_of_run,
                                                    self.start_time, self.line_item, self.forecast_id,
                                                    "Arima model couldnt converge using ES3",
                                                    name=errored_sku_table)
                                    df_fit, df_fcast, model_param_df = es_3(copy.deepcopy(model_input))

                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id, df_fit, df_fcast, model_param_df)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                model_param_final = model_param_final.append(model_param_df)
                                if f_period == min(self.fcast_periods):
                                    restart = False


                            elif self.best_model == NN or (self.best_model == NN_Xreg):
                                np.random.seed(123)
                                df_fit, df_fcast = nn(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id, df_fit, df_fcast)
                                df_fit, df_fcast = clean_dataframe(df_fit, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                df_final_fit = df_final_fit.append(df_fit)
                                # model_order_final = model_order_final.append(model_order_df)
                                if f_period == min(self.fcast_periods):
                                    restart = False

                            elif self.best_model == MA:
                                df_fcast = moving_average_function(copy.deepcopy(model_input), n=3)
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                if f_period == min(self.fcast_periods):
                                    restart = False


                            elif self.best_model == CROSTON:
                                df_fcast = croston_function(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                if f_period == min(self.fcast_periods):
                                    restart = False


                            elif self.best_model == TSB:
                                df_fcast = tsb_function(copy.deepcopy(model_input))
                                add_metadata(self.location, self.part_number, self.type_sku, f_period, self.var_names,
                                             self.line_item,
                                             self.training_time_of_run, self.start_time, self.forecast_id, df_fcast)
                                df_forecast = df_forecast.append(df_fcast)
                                if f_period == min(self.fcast_periods):
                                    restart = False

                            else:
                                df_forecast = use_moving_average_for_exception(f_period, df_forecast)
                                save_error_skus(self.part_number, self.location, self.best_model, self.training_time_of_run,
                                                self.start_time, self.line_item, self.forecast_id,
                                                "cannot find a pure times Series model, used moving average",
                                                name=errored_sku_table)
                                if f_period == min(self.fcast_periods):
                                    restart = False

                    except Exception:
                        df_forecast = use_moving_average_for_exception(self, f_period)
                        save_error_skus(self.part_number, self.location, self.best_model, self.training_time_of_run,
                                        self.start_time, self.line_item, self.forecast_id,
                                        "Train and test split error in runner, used moving avregae",
                                        name=errored_sku_table)
                        if f_period == min(self.fcast_periods):
                            restart = False



        model_param_final = model_param_final.astype(str)
        # Store Final Forecast in Database and Excels
        save_to_db(df_final_fit, kind=Fitted_data_table_name)
        save_to_db(df_forecast, kind=Forecast_table_name)
        save_to_db(model_param_final, kind=Model_parameter_table_name)
        # save_to_db(model_order_final, kind='Forecast_Model_Order')

        # save_to_excel(df_final_fit, self.group, self.location, self.line_item, name='fitted', period=period)
        # save_to_excel(df_forecast, self.group, self.location, self.line_item, name='test_forecast', period=period)
        # save_to_excel(coef_final, self.group, self.location, self.line_item, name='coefficients')
        # save_to_excel(model_order_final, self.group, self.location, self.line_item, name='model_order')

        print('*' * 50)
        print('End of Modeling for {}/{}'.format(self.location, self.part_number))
        print('*' * 50)

    # --------------------------------------------------------------------------------------------------------------------------------------------




def use_moving_average_for_exception(self, f_period):
    df_fcst = pd.DataFrame()
    train_data = self.df
    train_data["Date"] = train_data.index
    model_input = ModelInput(self.location, self.part_number, self.parameters, None,
                             train_data, None, f_period, None)
    df_fcast = moving_average_function(copy.deepcopy(model_input), n=3)

    if self.best_model == inactive:
        df_fcast[model_name] = inactive

    add_metadata(self.location, self.part_number, self.type_sku, f_period, None,
                 self.line_item, self.training_time_of_run, self.start_time, self.forecast_id, df_fcast)
    df_fcst = df_fcst.append(df_fcast)
    return df_fcst
