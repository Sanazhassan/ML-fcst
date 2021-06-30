# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:05:54 2020

@author: Sana Hassan
"""
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


def lasso_selection(data, variables, names):
    #    variables = variables.set_index('Date', drop=False)
    total_df = pd.concat([data['ship_qty'], variables], axis=1).dropna(axis=0)
    _, i = np.unique(total_df.columns, return_index=True)
    total_df = total_df.iloc[:, i]
    total_df = total_df.reset_index()
    #    Prepare Target data
    lasso_date = data.index.max()
    train_data = total_df[total_df['Date'] <= lasso_date][['Date', 'ship_qty']]
    train_data_scaled = preprocessing.scale(train_data['ship_qty'])
    #    Prepare variable data
    train_var = total_df[total_df['Date'] <= lasso_date]
    train_var = train_var[names]
    train_var_scaled = pd.DataFrame(preprocessing.scale(train_var), columns=names)

    #    Lasso
    lasso = Lasso(alpha=0.04, normalize=True, random_state=1234)
    lasso.fit(train_var_scaled, train_data_scaled)


lasso.feature_names = names
coeffs = pd.DataFrame()
coeffs['coef'] = lasso.coef_
coeffs['vars'] = lasso.feature_names
coeffs = coeffs[coeffs['coef'] != 0]
print("Number of relevant variables found: ", len(coeffs))  # (This tells you how many variables were selected)
print(coeffs[['coef', 'vars']])

final_var = coeffs['vars'].to_list()
#    final_var.append('order_qty')
#    final_var = ['order_qty']

final_var_df = variables[final_var]
final_var_df = final_var_df.reset_index(drop=False)
final_var_df = final_var_df.set_index('Date', drop=False)
return final_var, final_var_df
