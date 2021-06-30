# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:17:51 2020

@author: Sana Hassan
"""

import pandas as pd
from datetime import datetime

import pandas as pd
from datetime import datetime
from database.sql import *
from data_manager.table_names import *


class Parameters:

    division_filter = []
    start_date_filter = ''
    last_date_filter = ''
    forecast_start_date_filter = ''

    holdout_periods = []
    tsfreq = []

    fcast_periods = []
    parallelize = ''
    forecast_name = ''
    fast_mover_cutoff = []
    slow_mover_cutoff = []
    new_product_cutoff = []
    zero_demand_cutoff = []

    def __init__(self, path):
        self.file_path = path

    def read(self):
        df = pd.read_csv(self.file_path)
        for _, row in df.iterrows():
            self.tsfreq = 12  # change for a different frequency

            if row['Parameter_Type'] == 'Filter':
                if row['Parameter'] == 'division_filter':
                    filters = str.split(row['Value'], ',')
                    self.division_filter = [x.strip() for x in filters]
                elif row['Parameter'] == 'start_date_filter':
                    self.start_date_filter = datetime.strptime(row['Value'], '%m/%d/%Y')
                elif row['Parameter'] == 'training_last_date':
                    self.last_date_filter = datetime.strptime(row['Value'], '%m/%d/%Y')
                elif row['Parameter'] == 'forecast_start_date':
                    self.forecast_start_date_filter = datetime.strptime(row['Value'], '%m/%d/%Y')


            elif row['Parameter_Type'] == 'Rolling Forecasts':
                if row['Parameter'] == 'holdout_periods':
                    periods = str.split(str(row['Value']), ',')
                    periods = [x.strip() for x in periods]
                    self.holdout_periods = [int(x) for x in periods]
                elif row['Parameter'] == 'fcast_period':
                    periods = str.split(str(row['Value']), ',')
                    periods = [x.strip() for x in periods]
                    self.fcast_periods = [int(x) for x in periods]


            elif row['Parameter_Type'] == 'Ops':
                if row['Parameter'] == 'parallelize':
                    self.parallelize = (row['Value'])
                elif row['Parameter'] == 'forecast_name':
                    self.forecast_name = (row['Value'])
                elif row['Parameter'] == 'fast_mover_cutoff':
                    self.fast_mover_cutoff = float(row['Value'])
                elif row['Parameter'] == 'slow_mover_cutoff':
                    self.slow_mover_cutoff = float(row['Value'])
                elif row['Parameter'] == 'new_product_cutoff':
                    self.new_product_cutoff =  int(row['Value'])
                elif row['Parameter'] == 'zero_demand_cutoff':
                    self.zero_demand_cutoff = int(row['Value'])

    def read_only(self):
        df = pd.read_csv(self.file_path)
        return df

    def to_dataframe(self):
        df = self.df
        return df
