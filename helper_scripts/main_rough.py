# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:10:30 2020
@author: Sana Hassan
"""

import warnings
import os
from datetime import datetime
import argparse
from forecasting.runner import Runner
from multiprocessing import Process
from metrics.metrics import Metrics
# from data_manager.dependencies import download_packages
from data_manager.aggregation import aggregated_data, get_combinations
from data_manager.read_data import read_data


# from forecasting.forecast import Forecast


def main(_args):
    """
    A method to drop the tables if they exists, download packages, initiate
    the process of preparing the dataset for running the model for each product hierarchy combination
    """
    #    download_packages() #TODO automate the package check
    # drop_all_tables()
    metrics = Metrics()
    start_time = datetime.today().strftime('%Y-%m-%d-%H:%M')
    print("START TIME = {}".format(start_time))
    processes = []

    # reads data from parser
    data_raw = read_data(args)
    # data_raw = read_data_from_blob(args)
    data = data_raw
    #    data = aggregated_data(data_raw)

    # multiprocessing
    for rf in data.parameters.region_filters:
        if data.parameters.parallelize == 'Yes':
            p = Process(target=run_pipeline, args=(rf, data, metrics, start_time))
            p.start()
            processes.append(p)

        else:
            run_pipeline(rf, data, metrics, start_time)

    for p in processes:
        p.join()

    end_time = datetime.now()
    print("END TIME = {}".format(end_time))
    metrics.print()


#    print("TOTAL TIME TAKEN = {}s".format(end_time - start_time))


def run_pipeline(rf, data, metrics, start_time):
    # Runs the project pipeline
    metrics.add(rf, datetime.now())
    print("PROCESS ID = {}. region FILTER = {}".format(os.getpid(), rf))

    parameters = data.parameters

    #    get all combinations of the prodcut and location to loop through all products
    all_combinations = get_combinations(data.target)
    #    all_combinations = all_combinations[all_combinations["type"] =="slow mover"]
    #    all_combinations = all_combinations[1:7]

    # for each product (region)
    for index, row in all_combinations.iterrows():
        # subset data
        #        pf = "10643-16-12"
        #        rf = 8

        rf = row["warehouse_no"]
        pf = row["part_number"]

        print('*' * 50)
        print('Starting modeling proccess for location: {} and Product: {} at index {}'.format(rf, pf, index))
        print('*' * 50)

        data_subset, var_subset, var_names = data.process(rf, pf, start_time)

        # Initiate to run the models with prepared Data
        if len(data_subset) > 0:
            model_runner = Runner(data_subset, var_subset, var_names, parameters, rf, pf, metrics, start_time)

            model_runner.run()

            # model_forecast = Forecast(data_subset, var_subset, var_names, parameters, rf, pf, metrics, start_time)

            # model_forecast.run_forecast()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # needs to be changed according to the data input
    target_loc = "https://ddmaidevstorageac01.blob.core.windows.net/ddmai-dev-training/data/hpd_sample_data_processed.csv"
    para_loc = 'https://ddmaidevstorageac01.blob.core.windows.net/ddmai-dev-training/data/master_parameter.csv'
    var_loc = "https://ddmaidevstorageac01.blob.core.windows.net/ddmai-dev-training/data/variables.csv"

    parser = argparse.ArgumentParser(description='run forecasting for given data')
    parser.add_argument('--target', default=target_loc)
    parser.add_argument('--parameters', default=para_loc)
    parser.add_argument('--variables', default=var_loc)
    parser.add_argument('--parallelize', default=False)

    args = parser.parse_args(args=[])
    main(args)
    # print(args)
