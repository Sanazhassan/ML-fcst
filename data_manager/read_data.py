from data_manager.parameters import Parameters
from database.sql import *
from data_manager.data import Data
import pandas as pd
from data_manager.data_profile import data_profile


def read_data(args):
    return parse_input(args)


def read_and_clean(file_path):
    df = pd.read_csv(file_path, engine='python')
    # change date to YYYY-mm-dd
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df.Date.apply(lambda x: x.date().strftime('%Y-%m-%d'))
    return df


def clean(df):
    df['Date'] = pd.to_datetime(df['Date'])
    # df['Date'] = df.Date.apply(lambda x: x.date().strftime('%Y-%m-%d'))
    return df


def parse_input(_args):
    """
    Argument parser
    """
    parameters = Parameters(_args.parameters)
    parameters.read()
    variables_data = read_and_clean(_args.variables_data)
    target = get_all(_args.target)  # to read from Azure SQl
    target = clean(target)
    best_variables = get_recent_training_result(_args.best_variables_data)  # to read from Azure SQl
    best_models = get_recent_training_result(_args.best_model_data)
    return Data(parameters, variables_data, target, best_variables, best_models)
