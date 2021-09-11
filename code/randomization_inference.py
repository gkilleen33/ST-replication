# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
from patsy import dmatrices
from functools import partial


def ri_design_matrices(dataframe, model, time_name):
        dataframe = dataframe.select_dtypes(['number'])
        model = model + ' -1'  # Get rid of intercept since the data will be demeaned
        _y_df, _x_df = dmatrices(model, dataframe, 1, NA_action='raise', return_type='dataframe')
        _x_df[time_name] = dataframe[time_name]
        _y_df[time_name] = dataframe[time_name]
        for var in _x_df.columns:
            if var != time_name:
                _x_df[var] = _x_df[var] - _x_df.groupby(time_name)[var].transform('mean')
        for var in _y_df.columns:
            if var != time_name:
                _y_df[var] = _y_df[var] - _y_df.groupby(time_name)[var].transform('mean')
        _x_df.drop(columns=[time_name], inplace=True)
        _y_df.drop(columns=[time_name], inplace=True)
    
        _x = _x_df.to_numpy(dtype=np.float64)
        _y = _y_df.to_numpy(dtype=np.float64)
        return _y, _x


# Function to calculate randomization inference p-values (does not account for spatial correlation)
def ri_iteration(state, model, data, entities, number_slave_trade, number_persistent):
    df_copy = data.drop(columns=['persistent', 'slave_trade'])
        
    # Randomly assign persistence
    df_entities = pd.DataFrame()
    df_entities['ethnicity_id'] = entities
    df_entities = df_entities.sample(frac=1, random_state=state, replace=False).reset_index(drop=True)
    df_entities['persistent'] = 0 
    df_entities.loc[:number_persistent, 'persistent'] = 1
    
    # Randomly assign slave trade 
    df_entities = df_entities.sample(frac=1, random_state=state, replace=False).reset_index(drop=True)
    df_entities['slave_trade'] = 0 
    df_entities.loc[:number_slave_trade, 'slave_trade'] = 1
                            
    # Now replace persistent and slave_trade with random assignments in the data
    df_copy = df_copy.merge(df_entities, how='left', on='ethnicity_id', validate='many_to_one')
    
    # Estimate the regression 
    _y, _x = ri_design_matrices(dataframe=df_copy, model=model, time_name='year')
    _beta = np.dot(np.linalg.inv(np.dot(_x.T, _x)), (np.dot(_x.T, _y)))
    return [state, list(_beta)]


def ri_p_values(model, data, params, iterations=1000):
    # Get share of entities for which persistent is 1
    df_entities = data.drop_duplicates(subset=['ethnicity_id'], keep='first', ignore_index=True)
    number_persistent = df_entities['persistent'].sum()
    number_slave_trade = df_entities['slave_trade'].sum()
        
    entities = data['ethnicity_id'].unique()    
    
    # Generate a list to store the results from each run 
    ri_results = pd.DataFrame(columns = list(params.keys()))
        
    partial_function = partial(ri_iteration, model=model, 
                               data=data, entities=entities, 
                               number_slave_trade=number_slave_trade, 
                               number_persistent=number_persistent)
    
    
    for i in range(iterations):
        result = partial_function(i)
        ri_results.loc[result[0]] = result[1]
                           
    ri_p = dict()
    
    j = 0
    for var in list(params.keys()):
        num_larger = ri_results[abs(ri_results[var]) > abs(params[var])].count()
        p = num_larger/iterations
        ri_p[var] = p[0]
        j += 1
        
    return ri_p


## The above functions are for panel regressions, create a function for simple cross-sectional 
def ri_iteration_cs(state, model, data):
    df_copy = data.copy(deep=True)
        
    # Randomly shuffle persistence and slave trade
    df_copy['persistent'] = np.random.RandomState(seed=state).permutation(df_copy.persistent.values)
    df_copy['slave_trade'] = np.random.RandomState(seed=state).permutation(df_copy.slave_trade.values)
    
    # Estimate the regression 
    _y, _x = dmatrices(model, df_copy, 1)
    _beta = np.dot(np.linalg.inv(np.dot(_x.T, _x)), (np.dot(_x.T, _y)))
    return [state, list(_beta)]


def cs_ri_p_values(model, data, params, iterations=1000):
    # Generate a list to store the results from each run 
    ri_results = pd.DataFrame(columns = list(params.keys()))
        
    partial_function = partial(ri_iteration_cs, model=model, data=data)
    
    
    for i in range(iterations):
        result = partial_function(i)
        ri_results.loc[result[0]] = result[1]
                           
    ri_p = dict()
    
    j = 0
    for var in list(params.keys()):
        num_larger = ri_results[abs(ri_results[var]) > abs(params[var])].count()
        p = num_larger/iterations
        ri_p[var] = p[0]
        j += 1
        
    return ri_p
