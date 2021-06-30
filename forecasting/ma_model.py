"""
Created on Fri Oct 30 18:42:52 2020

@author: Sana Hassan

"""

from pandas.tseries.offsets import DateOffset
import pandas as pd
from forecasting.utils import forecast_df

VAL_COL = 'ship_qty'
MA = "MA"

def moving_average(train_df, n):
    average = sum(train_df[VAL_COL].tail(n)) // n
    return average


# Moving average: roll forward cross validation
def moving_average_function(arg, n):
    df_forecast = pd.DataFrame()
    for i in range(1, (arg.hp + 1)):
        y_pred = pd.Series()
        y_pred = pd.Series(moving_average(arg.train_data, n))
        y_pred.index = arg.train_data.tail(1).index + DateOffset(months=1)
        arg.train_data.loc[-1] = [arg.train_data["type"][0], arg.train_data["Part_number"][0],
                                  arg.train_data["Location"][0],
                                  y_pred.values[0], y_pred.index[0]]
        arg.train_data.index = arg.train_data["Date"]
        fcast = forecast_df(arg.parameters, y_pred, None, None, "MA")
        df_forecast = df_forecast.append(fcast)

    # df_fit = forecast_df(arg.parameters, None, None, None, "MA")
    return df_forecast
