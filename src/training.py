import numpy as np
from src.decoders import WienerFilter, KalmanFilter, FeedforwardNetwork, GRU
from bayes_opt import BayesianOptimization
from src.utils import partition, compute_R2

################################################################################

def train(S, Z, condition, method, config, optimize_flag):

    """
    Train a neural decoder either using preset hyperparameters
    or by learning hyperparameters using Bayesian optimization.

    Inputs
    ------
    S: list (1 x number of training trials) of N x T numpy arrays
        Each element of the list contains spiking data for N neurons over T times

    Z: list (1 x number of training trials) of M x T numpy arrays
        Each element of the list contains behavioral data for M behavioral variables over T times
    
    condition: list (1 x number of training trials) of condition IDs

    method: string indicating the neural decoder to use

    config: dictionary containing decoder settings

    optimize_flag: boolean indicating whether to use Bayesian optimization to learn hyperparameters
        If True, the settings for running the Bayesian optimization will be pulled from config
        If False, preset choices for hyperparameters will be pulled from config

    Outputs
    -------
    model: trained decoder object for the specified method

    HyperParams: dictionary containing the hyperparameters used to train model
   
    """

    # Optimize hyperparameters if desired.
    if optimize_flag:

        # Split data into training and validation sets.
        train_idx, val_idx = partition(condition, config['opt']['val_frac'])
        S_train = [S[i] for i in train_idx]
        S_val   = [S[i] for i in val_idx]
        Z_train = [Z[i] for i in train_idx]
        Z_val   = [Z[i] for i in val_idx]

        # Optimize hyperparameters.
        HyperParams = optimize_hyperparams(S_train, S_val, Z_train, Z_val, method, config['general'], config['opt'])

    else:

        # Unpack hyperparameters directly from config.
        HyperParams = config['general'].copy()
        beh_key = list(config.keys())
        beh_key.remove('general')
        beh_key.remove('opt')
        HyperParams.update(config[beh_key[0]])

    # Train model with chosen hyperparameters.
    model = train_specific_model(S, Z, method, HyperParams)

    return model, HyperParams

################################################################################

def train_specific_model(S, Z, method, HyperParams):

    """
    Train a specific type of neural decoder using
    a specific set of hyperparameters.

    Inputs
    ------
    S: list (1 x number of training trials) of N x T numpy arrays
        Each element of the list contains spiking data for N neurons over T times

    Z: list (1 x number of training trials) of M x T numpy arrays
        Each element of the list contains behavioral data for M behavioral variables over T times

    method: string indicating the neural decoder to use

    HyperParams: dictionary of hyperparameters

    Outputs
    -------
    model: trained decoder object for the specified method
   
    """

    # Train model for appropriate method.
    if method == 'wf':
        
        # Fit a Wiener filter.
        model = WienerFilter(HyperParams)
        model.fit(S, Z)

    elif method == 'kf':

        # Fit a Kalman filter.
        model = KalmanFilter(HyperParams)
        model.fit(S, Z)
    
    elif method == 'ffn':

        # Fit a feedforward neural network.
        model = FeedforwardNetwork(HyperParams)
        model.fit(S, Z)

    elif method == 'gru':

        # Fit a GRU.
        model = GRU(HyperParams)
        model.fit(S, Z)

    else:
        raise NameError('Method not recognized.')

    return model

################################################################################

def optimize_hyperparams(S_train, S_val, Z_train, Z_val, method, gen_hp, opt_config):

    """
    Learn a good set of hyperparameters to use with a particular
    neural decoding method by using Bayesian optimization.

    Inputs
    ------
    S_train: list (1 x number of training trials) of N x T numpy arrays
        Each element of the list contains spiking data for N neurons over T times

    S_val: list (1 x number of validation trials) of N x T numpy arrays
        Each element of the list contains spiking data for N neurons over T times

    Z_train: list (1 x number of training trials) of M x T numpy arrays
        Each element of the list contains behavioral data for M behavioral variables over T times

    Z_val: list (1 x number of validation trials) of M x T numpy arrays
        Each element of the list contains behavioral data for M behavioral variables over T times

    method: string indicating the neural decoder to use

    gen_hp: dictionary of hyperparameters that are set and won't be optimized

    opt_config: dictionary specifying details of Bayesian optimization
        This contains general Bayesian optimization settings and 
        ranges to search for hyperparameters that will be optimized.

    Outputs
    -------
    HyperParams: dictionary of good hyperparameters learned via Bayesian optimization
   
    """

    def evaluate_model(**kwargs):
        
        # Create dictionary of hyperparameters.
        HyperParams = construct_hyperparams(kwargs, gen_hp, method)

        # Train model.
        model = train_specific_model(S_train, Z_train, method, HyperParams)
        
        # Make predictions on validation set.
        Z_val_hat = model.predict(S_val)
        
        # Return mean R2 across decoded variables.
        tau = HyperParams['Delta']*(HyperParams['tau_prime']+1)-1
        return np.mean(compute_R2(Z_val, Z_val_hat, skip_samples=tau, eval_bin_size=5))
    
    # Unpack optimization settings.
    init_points = opt_config['init_points']
    n_iter = opt_config['n_iter']
    kappa = opt_config['kappa']

    # Reformat opt_config to store parameter bounds as tuples.
    pbounds = opt_config.copy()
    pbounds.pop('init_points')
    pbounds.pop('n_iter')
    pbounds.pop('kappa')
    pbounds.pop('val_frac')
    pbounds = {k: tuple(v) for k, v in pbounds.items()}

    # Optimize hyperparameters using the training and validation sets.
    optimizer = BayesianOptimization(evaluate_model, pbounds, verbose=1)
    optimizer.maximize(init_points=init_points, n_iter=n_iter, kappa=kappa)
    best_params = optimizer.max['params']
    HyperParams = construct_hyperparams(best_params, gen_hp, method)

    return HyperParams

################################################################################

def construct_hyperparams(optimized_hp, gen_hp, method):

    """
    Construct a unified hyperparameters dictionary.

    Inputs
    ------
    optimized_hp: dictionary of hyperparameters that were
        optimized (or are in the process of being optimized)

    gen_hp: dictionary of general hyperparameters that
        were not optimized

    method: string indicating the neural decoder

    Outputs
    -------
    HyperParams: dictionary of hyperparameters
   
    """

    # Initialize with general hyperparameters.
    HyperParams = gen_hp.copy()

    # Add method-specific hyperparameters.
    if method == 'wf':
        HyperParams['lam'] = optimized_hp['lam']
    elif method == 'kf':
        HyperParams['lag'] = int(np.round(optimized_hp['lag']/HyperParams['Delta'])*HyperParams['Delta']) # closest multiple of Delta
    elif method == 'ffn':
        HyperParams['num_units'] = int(optimized_hp['num_units'])
        HyperParams['num_layers'] = int(optimized_hp['num_layers'])
        HyperParams['frac_dropout'] = float(optimized_hp['frac_dropout'])
        HyperParams['num_epochs'] = int(optimized_hp['num_epochs'])
    elif method == 'gru':
        HyperParams['num_units'] = int(optimized_hp['num_units'])
        HyperParams['frac_dropout'] = float(optimized_hp['frac_dropout'])
        HyperParams['num_epochs'] = int(optimized_hp['num_epochs'])
    else:
        raise NameError('Method not recognized.')

    return HyperParams
