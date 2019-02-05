import numpy as np
import pandas as pd


def log_normalize_dist(data, column_names=[]):
    '''
       Normalize distribution by applytin log to all values.
       
       NOTE: ALL VALUES MUST BE POSSITIVE DEFINED!!!
       
        args:
            - data: Pandas DataFrame with values to normalize
            - column_names: Indicates column' names of the columns from 'data' to normalize
                * default value: "all columns"
        returns:
            - a new data frame with normalized columns (for the sake of understanding 'log_' is added as prefix to the
            original column name)
    '''
    # If no column names are specified apply normalization to all columns
    if len(column_names) == 0:
        column_names = data.columns.values

    norm_data = {}
    for cn in column_names:
        norm_data['log_' + cn] = np.log(data[cn])

    return pd.DataFrame(norm_data)


def get_outlier_bounds(data):
    '''
    Computes the lower and upper bound to catalog a point as an outlier
        args:
            - data: a Pandas Series with data from which outliers' bound should be computed
        return:
            - lower_bound, upper_bound: Both are float values containing the lower and upper bounds respectively
    '''
    # 1.5 the IQR as limits
    q3 = data.quantile(q=.75)
    q1 = data.quantile(q=.25)
    iqr = q3-q1
    upper_bound = round(q3 + 1.5*iqr, 2)
    lower_bound = round(q1 - 1.5*iqr, 2)
    
    #quantile .91, .05, .95 and .99 as limits
    q99 = round(data.quantile(q=.99), 2)
    q95 = round(data.quantile(q=.95), 2)
    q05 = round(data.quantile(q=.05), 2)
    q01 = round(data.quantile(q=.01), 2)

    return lower_bound, upper_bound, q01, q05, q95, q99


def treat_outliers(data_in, column_names=[], value='mean', method='whiskers'):
    '''
        args:
            - data: Pandas DataFrame with values with outliers
            - column_names: Indicates column' names of the columns from 'data' to which outliers treatment must be applyied
                * default value: "all columns"
            - value: Value to replace outlier values.
                * mean: Apply mean value of the column to the outliers (default value)
                * remove: Remove outliers from column (applying NaN)
                * min_max_value: Replace outlier values with the maximum (or minimum) value within method's limits
            - method: method to treat outliers
                * whiskers: Apply 'value' to those values beyond boxplot's whiskers (default method)
                * 0199: Apply 'value' to those values lower than percentile 1 and greather than percentile 99 
                * 0595: Apply 'value' to those values lower than percentile 5 and greather than percentile 95 
        returns:
            - a new data frame with outliers treated for the desired columns
    '''
    data = data_in.copy(deep=True)

    # If no column names are specified apply outliers treatment to all columns
    if len(column_names) == 0:
        column_names = data.columns.values
    
    for col_name in column_names:
        col_data = data[col_name] #Obtain series to perform outliers filtering
                
        lower_limit, upper_limit, q01, q05, q95, q99 = get_outlier_bounds(col_data) #Obtain limits to filter outliers

        # By default whiskers are used as limits
        if method == '0595':
            lower_limit = q05
            upper_limit = q95
        elif method == '0199':
            lower_limit = q01
            upper_limit = q99
       
        if value == 'min_max_value':
            col_data.loc[col_data<lower_limit] = lower_limit
            col_data.loc[col_data>upper_limit] = upper_limit
        elif value == 'remove':
            col_data.loc[(col_data<lower_limit) | (col_data>upper_limit)] = np.nan
        else: 
            mean_value = col_data.loc[(col_data>=lower_limit) & (col_data<=upper_limit)].mean()
            col_data.loc[(col_data<lower_limit) | (col_data>upper_limit)] = mean_value #Default value's choice 
    
        data[col_name] = col_data
                

    return data
        
