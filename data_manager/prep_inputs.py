from data_manager.aggregation import get_combinations
import pandas as pd


def data_batch(data):
    all_combinations = get_combinations(data.target)
    batch_list = pd.DataFrame()
    return_list = pd.DataFrame()

    for comb in all_combinations.iterrows():
        batch_list.append(comb)
        if len(batch_list) == 100:
            return_list.append(batch_list)
            batch_list.clear()
            return return_list
