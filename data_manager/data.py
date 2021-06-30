import pandas as pd
from forecasting.utils import create_lags,  add_metadata
from database.sql import save_to_db


class Data:
    def __init__(self, parameters, variables, target, best_variables, best_models):

        self.parameters = parameters
        self.variables = variables
        self.target = target
        self.best_variables = best_variables
        self.best_models = best_models



    def process(self, rf, pf, best_model, selected_variables, start_time):
        parameters = self.parameters

        variables = self.variables
        shipment_data = self.target

        print(rf, pf)
        # change data for sales
        data_subset = self.filter_data(shipment_data, parameters.start_date_filter, parameters.last_date_filter,
                                       rf, pf)

        if len(selected_variables) > 0:
            variable_with_int_var, data_subset = self.add_orders_price(data_subset,
                                                                       variables)  # add internal variables to the variables mix
            selected_variables_for_combination = selected_variables["Selected_variables"].to_list()
            required_columns = ["Date"] + selected_variables_for_combination
            required_variables = variable_with_int_var[
                required_columns]  # only variables that were selected in training
            # lags variable data according to the lags specified in correlated var
            var_subset, var_names = create_lags(data_subset, required_variables, selected_variables)
            # create date seq to account for unavailable date time index
            # dates_seq = pd.date_range(parameters.last_date_filter, periods=max(parameters.fcast_periods), freq="MS")
            # var_subset_future = pd.merge(dates_seq.to_frame(), var_subset, how='left', left_index=True, right_index=True)
            # var_subset_future["Date"] = var_subset_future.index
            # var_subset_future = var_subset_future[var_subset.columns]

            return data_subset, var_subset, var_names

        else:
            data_subset = data_subset[["Date", "type", 'Part_number', 'Location', 'ship_qty']]
            data_subset = data_subset.sort_values(by='Date', ascending=True).set_index('Date', drop=True)
            return data_subset, None, None

    # TODO chcek if a varible has history to be used for the selected forecast horizon, if not remove the variabke from the mix
    # TODO use only the valid historical data (if NA in the datasubset or varibales subset in start of the TS remove NAs then)


    @staticmethod
    def filter_data(df, start_date, end_date, rf, pf):
        # filter the dataset by start date, region, product, setting up frequency and order the data
        df = df.query('warehouse_no== @rf & part_number == @pf')
        # for col_name, col_value in kwargs.items():
        #     df = df[df[col_name] == col_value]
        df["Date"] = pd.to_datetime(df["Date"])

        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        df = df[['Date', 'type', 'part_number', 'warehouse_no', 'ship_qty', 'order_qty', 'avg_order_price', 'oem_forecast']]
        df.columns = ['Date', 'type', 'Part_number', 'Location', 'ship_qty', 'order_qty', 'avg_order_price', 'oem_forecast']
        df['ship_qty'] = pd.to_numeric(df['ship_qty'])
        df['order_qty'] = pd.to_numeric(df['order_qty'])
        df = df.drop_duplicates(ignore_index=True)
        # df = df.sort_values(by='Date', ascending=True).set_index('Date', drop=True)
        return df


    @staticmethod
    def add_orders_price(data_subset, variables):
        variables["Date"] = pd.to_datetime(variables["Date"])
        full_df = data_subset.merge(variables, on=["Date"], how='outer')
        internal_variables = ['order_qty', 'avg_order_price', 'oem_forecast']
        external_variables = variables.columns.to_list()
        all_variables = internal_variables + external_variables
        all_variables_data = full_df[all_variables]
        # all_variables_data = all_variables_data.sort_values(by='Date', ascending=True).set_index('Date', drop=True)

        shipment_data = data_subset[["Date", "type", 'Part_number', 'Location', 'ship_qty']]
        shipment_data = shipment_data.sort_values(by='Date', ascending=True).set_index('Date', drop=True)

        return all_variables_data, shipment_data
