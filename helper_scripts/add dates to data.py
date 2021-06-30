# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:46:04 2021

@author: 963675
"""

import pandas as pd

fm = pd.read_csv("C:/Users/963675/Documents/Industrilization/codes/data/input/hpd_sample_data.csv")
monthdata = pd.read_csv("C:/Users/963675/Documents/Industrilization/codes/data/input/monthdata.csv")

fm_data = fm.merge(monthdata, on="uni_month")

# change date format
fm_data["Date"] = pd.to_datetime(fm_data["Date"])
fm_data["Date"] = fm_data.Date.apply(lambda x: x.date().strftime('%Y-%m-%d'))

fm_data = fm_data.drop(columns=['month', 'year', 'day'])

fm_data.to_csv("C:/Users/963675/Documents/Industrilization/codes/data/input/hpd_sample_data_processed.csv", index=False)
