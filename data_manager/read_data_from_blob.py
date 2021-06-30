from data_manager.parameters import Parameters

from data_manager.data import Data
from pathlib import Path
import pandas as pd


def read_data(args):
    return parse_input(args)


def read_and_clean(file_path):
    df = pd.read_csv(file_path, engine='python')
    # change date to YYYY-mm-dd
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df.Date.apply(lambda x: x.date().strftime('%Y-%m-%d'))
    return df


def parse_input(_args):
    """
    Argument parser
    """
    p = Parameters(_args.parameters)
    p.read()
    v = read_and_clean(_args.variables)
    t = read_and_clean(_args.target)

    return Data(p, v, t)


def read_from_blob():
    from io import StringIO
    params = _args.Parameters
    blobstring = blob_service.get_blob_to_text(container_name, params).content
    # df = pd.read_excel(StringIO(blobstring))
    wb = load_workbook(self.file_path, data_only=True)
    ws = (wb["Master_parameter"])
    data = ws.values
    cols = next(data)
    data1 = list(data)
    df = pd.DataFrame(data1, columns=cols)
