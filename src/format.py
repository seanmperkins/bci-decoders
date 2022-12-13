import numpy as np
import pickle
import yaml
import copy
import os
from src.utils import partition

################################################################################

def load_data(dataset, test_frac):

    """
    Load data set and split into training and testing sets.

    Inputs
    ------
    dataset: string containing filename of dataset (without .pickle extension)

    test_frac: fraction of trials in the data set that should be reserved for testing

    Outputs
    -------
    Train: dictionary containing trialized neural and behavioral data in training set

    Test: dictionary containing trialized neural and behavioral data in testing set
   
    """

    # Load data file.
    with open('data/' + dataset + '.pickle','rb') as f:
        Data = pickle.load(f)

    # Format condition.
    if 'condition' in Data.keys():
        Data['condition'] = list(Data['condition']) # convert from np array to list
    else:
        first_key = list(Data.keys())[0]
        n_trials = len(Data[first_key])
        Data['condition'] = [np.nan]*n_trials # if no condition key exists, make one with all NaNs

    # Partition into training and testing sets.
    train_idx, test_idx = partition(Data['condition'], test_frac)
    Train = dict()
    Test = dict()
    for k in Data.keys():
        Train[k] = [Data[k][i] for i in train_idx]
        Test[k] = [Data[k][i] for i in test_idx]
    
    return Train, Test

################################################################################

def save_data(Results, run_name):

    """
    Save decoding results.

    Inputs
    ------
    Results: dictionary containing decoding results

    run_name: filename to use for saving results (without .pickle extension)
   
    """

    # If the 'results' directory doesn't exist, create it.
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save Results as .pickle file.
    with open('results/' + run_name + '.pickle','wb') as f:
        pickle.dump(Results,f)
    
################################################################################

def restrict_data(Train, Test, var_group):

    """
    Restrict training and testing data to only include a particular behavioral variable group(s).

    Inputs
    ------
    Train: dictionary containing trialized neural and behavioral data in training set

    Test: dictionary containing trialized neural and behavioral data in testing set

    var_group: string (or list of strings) containing behavioral variable group(s) to restrict to

    Outputs
    -------
    Train_b: copy of Train where behavioral data has been restricted as specified by var_group

    Test_b: copy of Test where behavioral data has been restricted as specified by var_group
   
    """
    
    # Initialize outputs.
    Train_b = dict()
    Test_b = dict()
    
    # Copy spikes into new dictionaries.
    Train_b['spikes'] = copy.deepcopy(Train['spikes'])
    Test_b['spikes'] = copy.deepcopy(Test['spikes'])
    
    # Copy condition labels (if task had condition structure).
    if 'condition' in Train:
        Train_b['condition'] = copy.deepcopy(Train['condition'])
        Test_b['condition'] = copy.deepcopy(Test['condition'])
    else:
        Train_b['condition'] = []
        Test_b['condition'] = []
        
    # Copy over relevant behavioral variables (and concatenate them into a single variable).
    if isinstance(var_group,str): # if var_group is just one variable...
        Train_beh = copy.deepcopy(Train[var_group])
        Test_beh = copy.deepcopy(Test[var_group])
    elif isinstance(var_group,list): # if var_group is multiple variables...
        Train_beh = copy.deepcopy(Train[var_group[0]])
        Test_beh = copy.deepcopy(Test[var_group[0]])
        for i in range(1,len(var_group)):
            Train_beh = [np.vstack((tb,v)) for tb,v in zip(Train_beh,Train[var_group[i]])]
            Test_beh = [np.vstack((tb,v)) for tb,v in zip(Test_beh,Test[var_group[i]])]
    else:
        raise Exception('Unexpected type for var_group.')
    Train_b['behavior'] = Train_beh
    Test_b['behavior'] = Test_beh

    return Train_b, Test_b
    
################################################################################
    
def load_config(dataset):

    """
    Load config file for a particular dataset.

    Inputs
    ------
    dataset: string containing filename of config file (without .yml extension)

    Outputs
    ------
    config: dictionary containing settings related to the dataset for each method
   
    """

    # Load config file.
    config_path = 'config/' + dataset + '.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

################################################################################

def relevant_config(config, method, var_group):

    """
    Restrict config dictionary to the settings specific to a given method and variable group(s).

    Inputs
    ------
    config: dictionary containing settings related to the dataset for each method

    method: string specifying which method to get settings for

    var_group: string (or list of strings) containing behavioral variable group(s) to get settings for

    Outputs
    -------
    model_config: dictionary containing settings from config that are relevant
        to the specified method and variable group(s)
   
    """

    # Get the portion of the config dictionary relevant to the provided method and variable group.
    if isinstance(var_group,str):
        model_config = {key:config[method][key] for key in ['general','opt',var_group]}
    elif isinstance(var_group,list):
        vg_name = '-'.join(var_group)
        model_config = {key:config[method][key] for key in ['general','opt',vg_name]}
    else:
        raise Exception('Unexpected type for var_group.')
    
    return model_config

################################################################################

def store_results(R2, behavior, behavior_estimate, HyperParams, Results, var_group, Train):

    """
    Store coefficient of determination (R2), ground truth behavior, 
    decoed behavior, and hyperparameters in the Results dictionary
    under key(s) indicating the behavioral variable group(s) these
    results correspond to.

    Inputs
    ------
    R2: 1D numpy array of coefficients of determination

    behavior: list of M x T numpy arrays, each of which contains ground truth behavioral data for M behavioral variables over T times

    behavior_estimate: list of M x T numpy arrays, each of which contains decoded behavioral data for M behavioral variables over T times

    HyperParams: dictionary of hyperparameters

    Results: method- and dataset-specific dictionary to store results in

    var_group: string (or list of strings) containing behavioral variable group(s)
        these results are associated with

    Train: dictionary containing trialized neural and behavioral data in training set
        This only gets used to help determine which R2 values go with which behavioral variables.
   
    """

    # Store R2, behavior, decoded behavior, and HyperParams in Results with the appropriate key.
    if isinstance(var_group,str):
        Results[var_group] = dict()
        Results[var_group]['R2'] = R2
        Results[var_group]['behavior'] = behavior
        Results[var_group]['behavior_estimate'] = behavior_estimate
        Results[var_group]['HyperParams'] = HyperParams.copy()
    elif isinstance(var_group,list):
        i = 0
        for v in var_group:
            m = Train[v][0].shape[0]
            Results[v] = dict()
            Results[v]['R2'] = R2[i:i+m]
            Results[v]['behavior'] = [b[i:i+m,:] for b in behavior]
            Results[v]['behavior_estimate'] = [b[i:i+m,:] for b in behavior_estimate]
            Results[v]['HyperParams'] = HyperParams.copy()
            i += m
    else:
        raise Exception('Unexpected type for var_group.')
