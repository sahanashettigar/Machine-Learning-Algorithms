'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float


def get_entropy_of_dataset(df):
    # TODO
    total_count = df.shape[0]
    if(total_count == 0):
        return 0
    col_list = df.columns.to_list()
    target_values = df[col_list[len(col_list)-1]].unique()
    target_count = np.unique(
        df[col_list[len(col_list)-1]], return_counts=True)[1]
    entropy = 0
    for i in range(len(target_count)):
        division_val = float(target_count[i]/sum(target_count))
        entropy += division_val*np.log2(division_val)
    entropy = -entropy
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float


def get_avg_info_of_attribute(df, attribute):
    # TODO
    total_count = df.shape[0]
    col_list = df.columns.to_list()
    if attribute not in col_list:
        return 0  # if attribute not in list then return 0
    #array_unique_values = df[attribute].unique()
    array_unique_values, unique_counts = np.unique(
        df[attribute], return_counts=True)
    avg_info = 0
    for i in range(len(array_unique_values)):
        #i_val_count = df[df[attribute] == i].shape[0]
        entropy_of_i = get_entropy_of_dataset(
            df[df[attribute] == array_unique_values[i]])
        if(entropy_of_i != 0):
            avg_info += float((unique_counts[i]/total_count))*entropy_of_i
    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float


def get_information_gain(df, attribute):
    # TODO
    col_list = df.columns.to_list()
    if attribute not in col_list:
        return 0
    information_gain = get_entropy_of_dataset(
        df)-get_avg_info_of_attribute(df, attribute)
    return information_gain


# input: pandas_dataframe
# output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    # TODO
    dict_info_gain = {}
    col_list = df.columns.to_list()
    col_list.pop()
    for i in col_list:
        dict_info_gain[i] = get_information_gain(df, i)
    max_info_gain_key = max(dict_info_gain, key=dict_info_gain.get)
    return (dict_info_gain, max_info_gain_key)
