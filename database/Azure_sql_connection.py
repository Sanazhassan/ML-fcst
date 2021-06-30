import pyodbc
import pandas as pd
import sqlalchemy

DB_NAME = '[Demand_Forecasting]'
# CONN = pyodbc.connect('DSN=Azure SQL database;UID=USPHC\963675;PWD=Summerheat2021', autocommit=True)  # TODO prompt for credentials
CONN = pyodbc.connect('DSN=Azure SQL database')  # TODO prompt for credentials

ENGINE = sqlalchemy.create_engine('mssql+pyodbc://Azure SQL database')


# query = 'select * from {};'.format(table_name)
# print(query)
# df = pd.read_sql(query, CONN)

def fully_qualified_name(name):
    return '{}.{}'.format(DB_NAME, name)


def get_all(name):
    table_name = fully_qualified_name(name)
    query = 'select * from {};'.format(table_name)
    print(query)
    return pd.read_sql(query, CONN)


def get_data_from_sql(table):
    data = get_all(table)
    return data
#    print(data)
