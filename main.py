"""
Created on Sat Feb  8 22:10:30 2020
@author: Sana Hassan
"""

# TODO cleanup imports that are not being used
import warnings
import os
import logging
import io
from datetime import datetime
import argparse
import pandas as pd
from forecasting.models_profiler import ParallelRun, RunAllModels
from metrics.metrics import Metrics
from data_manager.aggregation import aggregated_data, get_combinations
from database.sql import *
from data_manager.read_data import read_data
from data_manager.data_profile import data_profile
from data_manager.outlier_profile import outlier_profile
from data_manager.table_names import *
# import warnings filter
from warnings import simplefilter
pd.set_option('display.max_columns', None) # show all columns in console
pd.set_option('display.width', None)

# python
import os
os.environ['MPLCONFIGDIR'] = '/tmp/'

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


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
    start_time_now = datetime.now()
    print("START TIME = {}".format(start_time))

    # reads data from parser
    data = read_data(args)
    # if  wants to run the engine with multiple Divisions put the below lines in a for loop
    # for i in data.parameters.division_filter:

    # add more info to parameters object
    data.parameters.training_time_of_run = data.best_models[["Training_Run_Date"]].values[0][0]
    data.parameters.forecast_id = data.best_variables[[fcast_id]].values[0][0]
    data.parameters.Forecast_Run_Date = start_time
    parameters = data.parameters
    


    # profile outliers
    data.target = outlier_profile(data.target)

    # profile data into fast medium slow movers
    data.target = data_profile(data.target, parameters.start_date_filter, parameters.last_date_filter,
                               parameters.fast_mover_cutoff, parameters.slow_mover_cutoff,
                               parameters.new_product_cutoff, parameters.zero_demand_cutoff)
    profiled_shipment_data = data.target
    ml_model_data = profiled_shipment_data[(profiled_shipment_data["type"] == fast_movers) | (profiled_shipment_data["type"] == medium_movers)]
    slow_mover_model_data = profiled_shipment_data[(profiled_shipment_data["type"] == slow_movers)]
    zero_demand_model_data = profiled_shipment_data[(profiled_shipment_data["type"] == zero_demand)]
    new_product_model_data = profiled_shipment_data[(profiled_shipment_data["type"] == new_product)]

    total_comb = get_combinations(profiled_shipment_data)
    print("total_combinations are:", total_comb.shape[0])

    ml_comb = get_combinations(ml_model_data)
    print("fast movers and medium movers combinations are:", ml_comb.shape[0])

    sm_zd_comb = get_combinations(slow_mover_model_data)
    print("slow movers  combinations are:", sm_zd_comb.shape[0])

    zd_comb = get_combinations(zero_demand_model_data)
    print("zero demand combinations are:", zd_comb.shape[0])

    np_comb = get_combinations(new_product_model_data)
    print("new product combinations are:", np_comb.shape[0])



    # run machine learning models on fast movers and medium movers data
    #----------------------------------------------------------------------------------------------------------------------#
    if len(ml_model_data) > 0:
        ParallelRun(ml_model_data, parameters, data, start_time) #TODO run in different processes (Low priority)

    end_time_1 = datetime.now()
    # ----------------------------------------------------------------------------------------------------------------------#
                                                #Slow mover a run
    # ----------------------------------------------------------------------------------------------------------------------#
    if len(slow_mover_model_data) > 0:
        ParallelRun(slow_mover_model_data, parameters, data, start_time)

    end_time_2 = datetime.now()

    # ----------------------------------------------------------------------------------------------------------------------#
    #                                                    New Product Runs
    # ----------------------------------------------------------------------------------------------------------------------

    if len(new_product_model_data) > 0:
        ParallelRun(new_product_model_data, parameters, data, start_time)

    end_time_3 = datetime.now()

    # ----------------------------------------------------------------------------------------------------------------------#
    #                                                      Zero demands
    # ----------------------------------------------------------------------------------------------------------------------

    if len(new_product_model_data) > 0:
        ParallelRun(zero_demand_model_data, parameters, data, start_time)

    end_time_4 = datetime.now()



    # ----------------------------------------------------------------------------------------------------------------------#
    #                                                     End of modeling
    # ----------------------------------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------------------------------#
    # Time metrics
    # ----------------------------------------------------------------------------------------------------------------------



    print('-' * 50)
    print("Finished forecasting module for fast movers and medium movers at = {}".format(end_time_1))
    metrics.print()
    print("TIME TAKEN for FM and MM= {}s".format(end_time_1 - start_time_now))

    print('-' * 50)
    print("Finished forecasting module for Slow movers  at = {}".format(end_time_2))
    metrics.print()
    print("TIME TAKEN for SM and ZP= {}s".format(end_time_2 - start_time_now))

    print('-' * 50)
    print("Finished forecasting module for New products at = {}".format(end_time_3))
    metrics.print()
    print("TIME TAKEN for NP= {}s".format(end_time_3 - start_time_now))

    print('-' * 50)
    print("Finished forecasting module for Zero demand at = {}".format(end_time_4))
    metrics.print()
    print("TIME TAKEN for NP= {}s".format(end_time_4 - start_time_now))

    print('-' * 50)
    end_time = datetime.now()
    print("Finished Full forecasting module at = {}".format(end_time))
    metrics.print()
    print("TOTAL TIME TAKEN = {}s".format(end_time - start_time_now))


    # ----------------------------------------------------------------------------------------------------------------------#
    #                                               Write Results to Excel
    # ----------------------------------------------------------------------------------------------------------------------
    # write_to_excel('dbo.{}'.format(Forecast_table_name), parameters)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    # needs to be changed according to the data input

    # phase 1 data
    target_loc = '[dbo].[phase2_analytical_dataset_oem]'

    # phase 2 data
    # target_loc = '[dbo].[final_analytical_dataset_fulldata]'

    # sample data 1000 combs
    # target_loc = '[gen1].[hpd_sample_data]'

    para_loc = "./data/master_parameter.csv"
    var_loc = "./data/input/variables.csv"
    best_model_loc = '[dbo].[best_models_0401]'
    best_variables_loc = '[dbo].[selected_variables_0401]'

    parser = argparse.ArgumentParser(description='run forecasting for given data')
    parser.add_argument('--target', default=target_loc)
    parser.add_argument('--parameters', default=para_loc)
    #parser.add_argument('--parameters')
    parser.add_argument('--variables_data', default=var_loc)
    parser.add_argument('--best_model_data', default=best_model_loc)
    parser.add_argument('--best_variables_data', default=best_variables_loc)
    parser.add_argument('--parallelize', default=True)

    args = parser.parse_args()
    main(args)











































#
#
#     # run machine learning models on fast movers and medium movers data
#     if len(ml_model_data) > 0:
#         ParallelRun(ml_model_data, parameters, data, start_time)  # TODO run in different processes (Low priority)
#     end_time_1 = datetime.now()
#
#
#     if len(slow_mover_model_data) > 0:
#         ParallelRun(slow_mover_model_data, parameters, data, start_time)
#     end_time_2 = datetime.now()
#
#
#     # Split datafarme into equal portions to run 45 processes
#     if len(new_product_model_data) > 0:
#         ParallelRun(new_product_model_data, parameters, data, start_time)
#
# # using batches for 45 process at a time -stoping and staring process after 45 process
# #     if len(new_product_model_data) > 0:
# #         FastMediumMoversRun(new_product_model_data, parameters, data, start_time)
#
#     end_time_3 = datetime.now()
#
# # Sequential run
#     # if len(new_product_model_data) > 0:
#     #     all_np_combinations = get_combinations(new_product_model_data)
#     #     MovingAverageRun(all_np_combinations, parameters, data, start_time)
#
#
#
#
#     # run machine learning models on fast movers and medium movers data
#     # if len(ml_model_data) > 0:
#     # TODO run in different processes (Low priority)
#     # if len(slow_mover_model_data) > 0:
#     #     SlowMoversRun(slow_mover_model_data, parameters, start_time)
#     # if len(new_product_model_data) > 0:
#     #     NewProductRun(new_product_model_data, parameters, start_time)