# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:41:06 2021
@author: Kiran N Sundaram
"""

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

# start_date_filter = datetime.datetime(2015,7, 1)
# last_date_filter = datetime.datetime(2020,12,1)

# sample data
# df = pd.read_csv("C:/Users/963904/Documents/Parker Demand Planning/Phase II/Working Codes/sample_profile_data.csv")
# use for sample data for import from flat file
# df['Date']=pd.to_datetime(df['Date'])


def data_profile(df, start_date_filter, last_date_filter, fast_mover_cutoff, 
                 slow_mover_cutoff, new_product_cutoff, zero_demand_cutoff):
    total_months = (last_date_filter.year - start_date_filter.year) * 12 + (
                last_date_filter.month - start_date_filter.month)

    # find first date for each part-locations
    nonzero_df = df.loc[df['ship_qty'] != 0]
    min_date = nonzero_df.groupby(['part_number', 'warehouse_no'])['Date'].min().reset_index()
    min_date.columns = ['part_number', 'warehouse_no', 'min_date']
    dates_added = pd.merge(df, min_date, how='left', on=['part_number', 'warehouse_no'])

    # find count of non-zero shipment values (only needed if count column is not included)
    counts = nonzero_df.groupby(['part_number', 'warehouse_no']).size()
    count_df = counts.to_frame(name='count_').reset_index()
    all_counts = pd.merge(dates_added, count_df, how='left', on=['part_number', 'warehouse_no'])
    
    # find ship quantities in the past year
    oneyeardata = all_counts.loc[(all_counts['Date']>(last_date_filter - relativedelta(months=zero_demand_cutoff))) 
                                 & (all_counts['Date']<=(last_date_filter))]
    oneyear = oneyeardata.groupby(['part_number', 'warehouse_no'])['ship_qty'].sum().reset_index()
    oneyear.columns = ['part_number', 'warehouse_no','sum']
    data = pd.merge(all_counts, oneyear, how='left', on=['part_number', 'warehouse_no'])
    data['sum'] = data['sum'].fillna(0)

    # find months active for all part-locations
    data['months_active'] = (last_date_filter.year - data['min_date'].dt.year) * 12 + (last_date_filter.month - data['min_date'].dt.month)
    data['months_active'] = data['months_active'].astype(float)

    # set conditions for profiling
    zerodemand = data['sum']==0
    newprod = data['months_active']<=new_product_cutoff
    fm = data['count_']>= fast_mover_cutoff*(total_months)
    mm = (data['count_']>=slow_mover_cutoff*(total_months))&(data['count_']<fast_mover_cutoff*(total_months))
    sm = data['count_']<=slow_mover_cutoff*(total_months)
    
    
    # profile data
    data['type'] = np.where(zerodemand, 'zero demand', 
                            np.where(newprod, 'new product',
                                     np.where(fm, 'fast mover', 
                                              np.where(mm, 'medium mover', 
                                                       np.where(sm, 'slow mover','NA')))))


    return data


