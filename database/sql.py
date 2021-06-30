# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:17:29 2020

@author: Sana Hassan
"""
import pyodbc
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import urllib
import time



DB_NAME = '[Demand_Forecasting]'
CONN = pyodbc.connect(
    'Driver={ODBC Driver 17 for SQL Server};Server=tcp:phc-slsmkt01-dev-sql.database.windows.net,'
    '1433;Database=Demand_Forecasting;Uid=sqlazureload;Pwd=yY4Cx8qnwN7YHmgy;Encrypt=yes;TrustServerCertificate=no'
    ';Connection Timeout=30;')  # TODO prompt for credentials

# CONN = pyodbc.connect('DSN=Azure SQL database')

# craete SQLalchemy engine from connection string
QUOTED = urllib.parse.quote_plus(
    "Driver={ODBC Driver 17 for SQL Server};Server=tcp:phc-slsmkt01-dev-sql.database.windows.net,"
    "1433;Database=Demand_Forecasting;Uid=sqlazureload;Pwd=yY4Cx8qnwN7YHmgy;Encrypt=yes;TrustServerCertificate=no"
    ";Connection Timeout=30;")
ENGINE = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(QUOTED), fast_executemany=True, pool_pre_ping=True)


# ENGINE = create_engine('mssql+pyodbc://Azure SQL database')


def fully_qualified_name(name):
    return '{}.{}'.format(DB_NAME, name)


def get_all(name):
    table_name = fully_qualified_name(name)
    query = 'select * from {};'.format(table_name)
    print(query)
    return pd.read_sql(query, CONN)


def get_recent_training_result(name):
    '''gets the most recent training results'''
    table_name = fully_qualified_name(name)
    query = 'SELECT * FROM {} where Training_Run_Date = (Select max(Training_Run_Date) FROM {});'.format(table_name, table_name)
    print(query)
    return pd.read_sql(query, CONN)



# def get_variables_data(name):
#     '''gets the most recent training results'''
#     table_name = fully_qualified_name(name)
#     geography = ''
#     query = 'SELECT * FROM {} where Training_Run_Date = (Select max(Training_Run_Date) FROM {});'.format(table_name, table_name)
#     print(query)
#     return pd.read_sql(query, CONN)





def save_to_db(df, kind):
    retry_flag = True
    retry_count = 0
    if len(df.index) == 0:
        print("Not saving to DB as {} is None".format(kind))
        return

    while retry_flag and retry_count < 5:
        try:
            df.to_sql('{}'.format(kind), ENGINE, if_exists='append', index=False, schema='dbo')
            retry_flag = False
        except:
            print ("Retry after 1 sec")
            retry_count = retry_count + 1
            time.sleep(1)


def write_to_excel(name, params):
    '''Write results in excel'''
    table_name = fully_qualified_name(name)
    query = 'SELECT * FROM {} where forecast_run_date = (Select max(forecast_run_date) FROM {});'.format(table_name, table_name)
    df = pd.read_sql(query, CONN)
    if len(df.index) == 0:
        print("Not saving to excel as {} is None".format(name))
        return
    file_name = "{}_{}_{}.xlsx".format(params.division_filter[0], name, params.forecast_run_date)
    writer = pd.ExcelWriter(file_name, engine='openpyxl')
    df.to_excel(writer, index=False)
    writer.save()

