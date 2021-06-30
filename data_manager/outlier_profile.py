# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 3:33:05 2021
@author: Kiran N Sundaram
"""
import pandas as pd

def outlier_profile(df):
    # get all nonzero shipment and orders values
    nonzero_ship_df = df.loc[df['ship_qty']!=0]
    nonzero_orders_df = df.loc[df['order_qty']!=0]

    # get the cut off for the top 2% of the ship and order quantity values
    ship_cutoff = nonzero_ship_df.groupby(['part_number','warehouse_no'])['ship_qty'].quantile(0.98).reset_index()
    ship_cutoff.columns=['part_number','warehouse_no','upper_ship_cutoff']
    nonzero_ship_df = pd.merge(nonzero_ship_df, ship_cutoff, on=['part_number','warehouse_no'], how="left")

    order_cutoff = nonzero_orders_df.groupby(['part_number','warehouse_no'])['order_qty'].quantile(0.98).reset_index()
    order_cutoff.columns=['part_number','warehouse_no','upper_order_cutoff']
    nonzero_orders_df = pd.merge(nonzero_orders_df, order_cutoff, on=['part_number','warehouse_no'], how="left")


    # get average of the ship and order quantity values
    ship_mean = nonzero_ship_df.groupby(['part_number','warehouse_no'])['ship_qty'].mean().reset_index()
    ship_mean.columns=['part_number','warehouse_no','ship_mean']
    nonzero_ship_df = pd.merge(nonzero_ship_df, ship_mean, on=['part_number','warehouse_no'], how="left")

    order_mean = nonzero_orders_df.groupby(['part_number','warehouse_no'])['order_qty'].mean().reset_index()
    order_mean.columns=['part_number','warehouse_no','order_mean']
    nonzero_orders_df = pd.merge(nonzero_orders_df, order_mean, on=['part_number','warehouse_no'], how="left")


    # replace all values above the top 2% with the mean
    nonzero_ship_df.loc[nonzero_ship_df['ship_qty']>nonzero_ship_df['upper_ship_cutoff'],'ship_qty'] = nonzero_ship_df['ship_mean']
    nonzero_ship_df = nonzero_ship_df[['Date','part_number','warehouse_no','ship_qty', 'order_qty']]

    nonzero_orders_df.loc[nonzero_orders_df['order_qty']>nonzero_orders_df['upper_order_cutoff'], 'order_qty'] = nonzero_orders_df['order_mean']
    nonzero_orders_df = nonzero_orders_df[['Date','part_number','warehouse_no','order_qty']]

    # join back with full data
    ship_subset = pd.merge(df, nonzero_ship_df,on=['Date','part_number','warehouse_no', 'order_qty'],how='left')
    ship_subset.loc[ship_subset['ship_qty_y'].isnull(), 'ship_qty_y']=ship_subset['ship_qty_x']
    data = pd.merge(ship_subset, nonzero_orders_df,on=['Date','part_number','warehouse_no'],how='left')
    data.loc[data['order_qty_y'].isnull(), 'order_qty_y']=data['order_qty_x']

    # rename dataframe columns
    data = data[['Date','part_number','warehouse_no','ship_qty_y','order_qty_y','avg_order_price', 'oem_forecast']]
    data.columns = ['Date','part_number','warehouse_no','ship_qty','order_qty','avg_order_price', 'oem_forecast']

    return data