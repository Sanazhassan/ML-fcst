# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:33:32 2020

@author: Sana Hassan
"""
import pandas as pd


def aggregated_data(data):
    # taken from master parameter file - this is changed manually
    print("frequency of input data:", data.parameters.freq_in)
    print("frequency desired:", data.parameters.freq_out)

    # if change is required (monthly to weekly or weekly to monthly)
    if (data.parameters.freq_in != data.parameters.freq_out):
        print("frequency  of input data and desired analysis if different")
        if (data.parameters.freq_out == 'monthly'):
            print(data.sales.head())
            data.sales = data.sales.set_index(["Date", 'Item', 'Geography', 'Product'], drop=False)
            data.variables = data.variables.set_index(["Date", 'Item', 'Geography', 'Product'], drop=False)
            #            df = data.sales.groupby(by = [pd.Grouper('Date', freq='M' ), pd.Grouper('Item')]).transform(foo)
            df_sales = data.sales.groupby(by=[pd.Grouper(level=0, freq="MS"), pd.Grouper(level=1), pd.Grouper(level=2),
                                              pd.Grouper(level=3)]).sum().reset_index()
            df_var = data.variables.groupby(
                by=[pd.Grouper(level=0, freq="MS"), pd.Grouper(level=1), pd.Grouper(level=2),
                    pd.Grouper(level=3)]).mean().reset_index()
            data.sales = df_sales
            data.variables = df_var

        else:
            data.sales = data.sales
            data.variables = data.variables
    print(data.sales.head())
    print(data.variables.head())
    return data


def get_combinations(data):
    all_combos = data.groupby(["part_number", "warehouse_no", "type"]).size().reset_index()
    all_combos = all_combos[["part_number", "warehouse_no", "type"]]
    # all_combos = all_combos.to_dict()
    return all_combos
