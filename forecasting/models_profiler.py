# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:10:30 2020
@author: Sana Hassan
"""

import warnings
import os
import logging
import io
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import traceback
from data_manager.aggregation import aggregated_data, get_combinations
from forecasting.runner import Runner
from multiprocessing import Process, Queue
from metrics.metrics import Metrics
# from data_manager.dependencies import download_packages
from data_manager.aggregation import aggregated_data, get_combinations
from forecasting.utils import save_error_skus, index_marks, split_dataframe
from pandas import DataFrame
from database.sql import save_to_db
# import warnings filter
from warnings import simplefilter
from data_manager.table_names import *

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

metrics = Metrics()
# start_time = datetime.today().strftime('%Y-%m-%d-%H:%M')
# start_time_now = datetime.now()
# print("START TIME = {}".format(start_time))
processes = []
best_models_per_batch = []
queue = Queue()

# initialized workers
workers = os.cpu_count() - 2


def ParallelRun(modeling_data, parameters, data, start_time):
    #    get all combinations of the part_number and location to loop through all part_numbers
    data.target = modeling_data
    all_combinations = get_combinations(modeling_data)
    # all_combinations = all_combinations[1:1000]
    print('skus to be trained in Slow and New Product category {}'.format(all_combinations.shape))
    workers = os.cpu_count() - 2
    chunk_rows = int(all_combinations.shape[0] / workers)
    # divide the  dataframe into equal datasets of 2000 combinations for parallel processing
    chunks = split_dataframe(all_combinations, chunk_rows)

    cols = all_combinations.columns
    select_comb = pd.DataFrame(columns=cols)
    batch_combinations = pd.DataFrame(columns=cols)
    errored_sku = pd.DataFrame(columns=cols)
    ma_data_with_index = modeling_data.set_index(["part_number", "warehouse_no", "type"], drop=True)

    for c in chunks:
        batch_combinations = c
        batch_combinations = batch_combinations.set_index(["part_number", "warehouse_no", "type"])
        batch_comb_data = ma_data_with_index[ma_data_with_index.index.isin(batch_combinations.index)].reset_index()
        batch_combinations = batch_combinations.reset_index()
        print(batch_combinations.shape)

        if parameters.parallelize == 'Yes':
            try:
                p = Process(target=RunAllModels, args=(batch_combinations, parameters, data, start_time))
                processes.append(p)
                p.start()

            except:
                traceback.print_exc()
                p.terminate()

        else:
            RunAllModels(batch_combinations, parameters, data, start_time)
            continue



    for p in processes:
        p.join()
        print("{}: ended".format(p))

    batch_combinations.drop(select_comb.index, inplace=True)
    processes.clear()


def run_pipeline(wn, pn, best_model, selected_variables, data_subset, var_subset, var_names, parameters, metrics,
                 start_time, queue):
    model_runner = Runner(best_model, selected_variables, data_subset, var_subset, var_names, parameters, wn, pn,
                          metrics, start_time)
    model_runner.run()


def RunAllModels(batch_combinations, parameters, data, start_time):
    for idx, row in batch_combinations.iterrows():

        wn = row["warehouse_no"]
        pn = row["part_number"]

        # wn = '687444' #order & ext var
        # pn = '10081-12-ZJ'

        print(wn)
        print(pn)


        best_model = data.best_models.query('Shipping_Location == @wn & Part_number == @pn')
        selected_variables = data.best_variables.query('Shipping_Location== @wn & Part_number == @pn')

        # debug various scenarios
        # best_model[["Best_model"]] = 'prophet_xreg'
        # selected_variables = selected_variables[selected_variables["Selected_variables"] != 'order_qty']
        # aa = pd.Series(["order_qty", 0.2, 0, '687542', '1XN77-20-16', None, "HPD", '2021-03-26-00:45', None, None],
        #                index=selected_variables.columns)
        # selected_variables = selected_variables.append(aa, ignore_index = True)
        #
        # selected_variables["Selected_variables"] = 'order_qty'


        if best_model[["Best_model"]].values[0][0] != 'error':

            try:
                data_subset, var_subset, var_names = data.process(wn, pn, best_model, selected_variables,
                                                                  start_time)
            except:
                traceback.print_exc()
                save_error_skus(pn, wn, best_model[["Best_model"]].values[0][0], parameters.training_time_of_run,
                                start_time, parameters.division_filter[0], parameters.forecast_id, "Error in subsetting the data for the part number and location, investigate data.py file", name=errored_sku_table)
                continue

            if len(data_subset) > 0:
                #     multiprocessing

                try:
                    print('Running  models for location: {} and part_number: {} at index {}'.format(wn, pn, idx))
                    run_pipeline(wn, pn, best_model, selected_variables, data_subset, var_subset, var_names,
                                 parameters, metrics, start_time, queue=None)
                except:
                    traceback.print_exc()
                    save_error_skus(pn, wn, best_model[["Best_model"]].values[0][0], parameters.training_time_of_run,
                                    start_time, parameters.division_filter[0], parameters.forecast_id, "Exception in run pipeline", name=errored_sku_table)
                    continue

            else:
                save_error_skus(pn, wn, best_model[["Best_model"]].values[0][0], parameters.training_time_of_run,
                                start_time, parameters.division_filter[0],  parameters.forecast_id, "couldn't find required data in input, data_subset is empty", name=errored_sku_table)

                continue


        else:
            save_error_skus(pn, wn, best_model[["Best_model"]].values[0][0], parameters.training_time_of_run,
                            start_time, parameters.division_filter[0],  parameters.forecast_id, "Training Best model is error, investigate issue in training run", name=errored_sku_table)
            continue
