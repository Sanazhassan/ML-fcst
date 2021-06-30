# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:17:27 2020

@author: Sana Hassan
"""

import pandas as pd

from dateutil.relativedelta import relativedelta


def data_split(data, var, var_names, period):
    """
    Splits the data and variables into three different datasets
    according to the time periods specified in the Parker_forecast_parameter.xlsx

    Takes historical and variables dataframe, variable names, period for forecast: default = 12,
    date where we need to start the validation from.

    Gives out three dataframe : Train , Validation and Test.
    """

    data["Date"] = data.index

    train_data = data

    if var_names is not None:
        train_var = var[(var['Date'] <= train_data['Date'].max())]
        test_var_full = var[(var['Date'] > train_data['Date'].max())]
        test_var_period_length = test_var_full.head(period)  # takes desired forecast period's variables data
        test_var = test_var_period_length.dropna(axis=1)
        test_var = test_var_period_length.set_index('Date', drop=True)
        if test_var.empty:
            train_var = None
            test_var = None

        else:
            train_var = train_var[test_var.columns]

    else:
        train_var = None
        test_var = None

    return train_data, train_var, test_var
