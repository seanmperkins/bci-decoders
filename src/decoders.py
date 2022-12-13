import numpy as np
from numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are
import tensorflow as tf
import tensorflow.keras.layers as layers
from src.utils import bin_spikes, append_history, array2list, zero_order_hold
import warnings

################################ WIENER FILTER #################################

class WienerFilter(object):

    """
    Class for the Wiener filter decoder

    Hyperparameters
    ---------------
    Delta: number of time points to pool into a time bin
    
    tau_prime: number of previous time bins (in addition to the current bin) to use for decoding
    
    lam: regularization parameter for ridge regression to learn weights
    
    """

    def __init__(self, HyperParams):
        self.Delta = HyperParams['Delta']
        self.tau_prime = HyperParams['tau_prime']
        self.lam = HyperParams['lam']

    def fit(self, S, Z):
    
        """
        Train Wiener filter.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Z: list of M x T numpy arrays, each of which contains behavioral data for M behavioral variables over T times
            
        Parameters
        ----------
        L: numpy array (N*(tau_prime+1)+1 x M) of weights
            Each column contains the weights for a different behavioral variable.
            The rows contain weights for each neuron and time bin over a recent history.
                There is one additional row to allow for a constant offset.
            
        """

        # Unpack attributes.
        Delta = self.Delta
        tau_prime = self.tau_prime
        lam = self.lam

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]

        # Reformat observations to include recent history.
        X = [append_history(s, tau_prime) for s in S]

        # Downsample kinematics to bin width and transpose.
        Z = [z[:,Delta-1::Delta].T for z in Z]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        X = [x[:,tau_prime:,:] for x in X]
        Z = [z[tau_prime:,:] for z in Z]

        # Flatten spike count tensor.
        X = [np.moveaxis(x, [0, 1, 2], [2, 0, 1]) for x in X]
        X = [x.reshape(x.shape[0], (x.shape[1]*x.shape[2])) for x in X]
        
        # Concatenate across trials.
        X = np.concatenate(X)
        Z = np.concatenate(Z)
        
        # Append column of ones to X to allow for constant offset.
        X = np.hstack((X, np.ones((X.shape[0],1))))
        
        # Train model.
        I = np.eye(X.shape[1])
        L = inv(X.T @ X + lam*I) @ X.T @ Z
        
        # Save parameters.
        self.L = L

    def predict(self, S):
    
        """
        Predict behavior with trained Wiener filter.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Outputs
        -------
        Z_hat: list of M x T numpy arrays, each of which contains decoded behavioral data for M behavioral variables over T times
   
        """

        # Unpack attributes.
        Delta = self.Delta
        tau_prime = self.tau_prime
        L = self.L

        # Compute trial lengths.
        T = [s.shape[1] for s in S]

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]

        # Reformat observations to include recent history.
        X = [append_history(s, tau_prime) for s in S]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        X = [x[:,tau_prime:,:] for x in X]

        # Flatten spike count tensor.
        X = [np.moveaxis(x, [0, 1, 2], [2, 0, 1]) for x in X]
        X = [x.reshape(x.shape[0], (x.shape[1]*x.shape[2])) for x in X]
        
        # Append column of ones to S to allow for constant offset.
        X = [np.hstack((X_mat, np.ones((X_mat.shape[0],1)))) for X_mat in X]
        
        # Predict behavior linearly.
        Z_hat = [X_mat @ L for X_mat in X]
        Z_hat = [Z.T for Z in Z_hat]

        # Add NaNs where predictions couldn't be made due to insufficient spiking history.
        Z_hat = [np.hstack((np.full((Z.shape[0],tau_prime), np.nan), Z)) for Z in Z_hat]

        # Return estimate to original time scale.
        Z_hat = [zero_order_hold(Z,Delta) for Z in Z_hat]
        Z_hat = [np.hstack((np.full((Z.shape[0],Delta-1), np.nan), Z)) for Z in Z_hat]
        Z_hat = [z[:,:t] for z,t in zip(Z_hat, T)]
        
        return Z_hat

################################ KALMAN FILTER #################################

class KalmanFilter(object):

    """
    Class for the Kalman filter decoder

    Hyperparameters
    ---------------
    Delta: number of time points to pool into a time bin
    
    lag: number of time samples to lag behavioral variables relative to spiking data
        This accounts for the physiological latency between when neurons become active
        and when that activity impacts behavior.
    
    steady_state: boolean determining whether the steady-state form of the Kalman filter should be used
        The steady-state Kalman filter is much faster for prediction, but takes
        a few samples to converge to the same output as the standard Kalman filter.
    
    """

    def __init__(self,HyperParams):
        self.Delta = HyperParams['Delta']
        self.lag = HyperParams['lag']
        self.bin_lag = int(np.round(self.lag / self.Delta))
        self.Ts = .001 # hardcodes sampling period at 1 ms
        self.dt = self.Delta * self.Ts
        self.steady_state = HyperParams['steady_state']

    def fit(self, S, Z):

        """
        Train Kalman filter.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Z: list of 4 x T numpy arrays, each of which contains behavioral data for position and velocity
            The first two rows correspond to x- and y-position
            The last two rows correspond to x- and y-velocity
            
        Parameters (Steady-state Kalman filter)
        ----------
        A: state-transition matrix (7 x 7 numpy array)
        C: observation matrix (N x 7 numpy array)
        K_inf: steady-state Kalman gain (7 x N numpy array)
        z0: initial state mean (7 x 1 numpy 1D array)
        
        Parameters (Standard Kalman filter)
        ----------
        A: state-transition matrix (7 x 7 numpy array)
        C: observation matrix (N x 7 numpy array)
        Q: state-transition noise covariance (7 x 7 numpy array)
        R: observation noise covariance (N x N numpy array)
        z0: initial state mean (7 x 1 numpy 1D array)
        P0: initial state covariance (7 x 7 numpy array)
            
        """

        # Unpack attributes.
        Delta = self.Delta
        lag = self.lag
        dt = self.dt
        
        # Shift data to account for lag.
        if self.lag > 0:
            S = [s[:,:-lag] for s in S]
            Z = [z[:,lag:] for z in Z]

        # Compute acceleration. We are assuming that the first two components of Z are something
        # akin to position and the second two components are the something akin to velocity.
        acc = [np.hstack((np.zeros((2,1)),np.diff(z[2:4,:],axis=1)/self.Ts)) for z in Z]
        Z = [np.vstack((z,a)) for z,a in zip(Z,acc)]
        
        # Add a row of ones to allow for constant offset.
        Z = [np.vstack((z,np.ones(z.shape[1]))) for z in Z]
        
        # Downsample kinematics.
        Z = [z[:,Delta-1::Delta] for z in Z]

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]
        
        # Convert lists (one entry per trial) to large arrays.
        Z_init = np.concatenate([np.reshape(z[:,0], (-1, 1)) for z in Z], axis=1)
        Z1 = np.concatenate([z[:,:-1] for z in Z], axis=1)
        Z2 = np.concatenate([z[:,1:] for z in Z], axis=1)
        Z = np.concatenate(Z, axis=1)
        S = np.concatenate(S, axis=1)
        
        # Fit state transition matrix.
        A = Z2 @ Z1.T @ inv(Z1 @ Z1.T)
        
        # Fit measurement matrix.
        C = S @ Z.T @ inv(Z @ Z.T)
        
        # Fit state noise covariance.
        T1 = Z1.shape[1]
        Q = ((Z2 - A @ Z1) @ (Z2 - A @ Z1).T) / T1
        
        # Fit measurement noise covariance.
        T2 = Z.shape[1]
        R = ((S - C @ Z) @ (S - C @ Z).T) / T2
        
        # Fit initial state
        z0 = np.mean(Z_init, axis=1)
        P0 = np.cov(Z_init, bias=True)
        
        # Store parameters appropriate for standard or steady-state Kalman filter.
        if self.steady_state:
            
            try:
                # Compute steady-state Kalman gain.
                Q = (Q + Q.T)/2 # ensures Q isn't slightly asymmetric due to floating point errors before running 'dare'
                R = (R + R.T)/2 # ensures R isn't slightly asymmetric due to floating point errors before running 'dare'
                P_inf = solve_discrete_are(A.T,C.T,Q,R)
                K_inf = P_inf @ C.T @ pinv(C @ P_inf @ C.T + R)

                # Store parameters for steady-state Kalman filter.
                self.A = A
                self.C = C
                self.K_inf = K_inf
                self.z0 = z0

            except np.linalg.LinAlgError:

                # The 'solve_discrete_are' function won't always succeed due
                # to numerical properties of the matrices that get fed into
                # it. When it fails, issue a warning to the user letting them
                # know we'll have to revert to the standard Kalman filter.
                warn_str = '''
                Discrete-time algebraic Riccati equation could not be solved using the learned parameters. 
                Reverting from steady-state Kalman filter back to standard Kalman filter.'''
                warnings.warn(warn_str)

                # Revert to standard Kalman filter.
                self.steady_state = False
                self.A = A
                self.C = C
                self.Q = Q
                self.R = R
                self.z0 = z0
                self.P0 = P0

        else:

            # Store parameters for standard Kalman filter.
            self.A = A
            self.C = C
            self.Q = Q
            self.R = R
            self.z0 = z0
            self.P0 = P0

    def predict(self, S):

        """
        Predict behavior with trained Kalman filter.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Outputs
        -------
        Z_hat: list of 4 x T numpy arrays, each of which contains decoded position and velocity
            The first two rows correspond to x- and y-position estimates
            The last two rows correspond to x- and y-velocity estimates
   
        """

        # Unpack attributes.
        Delta = self.Delta
        bin_lag = self.bin_lag
        A = self.A
        C = self.C
        z0 = self.z0
        if self.steady_state:
            K_inf = self.K_inf
        else:
            Q = self.Q
            R = self.R
            P0 = self.P0

        # Compute trial lengths.
        T = [s.shape[1] for s in S]

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]
        
        # Create estimates for each trial.
        n_trials = len(S)
        Z_hat = [None] * n_trials
        for tr in range(n_trials):
            
            # Initialize Z_hat for this trial.
            M = A.shape[0]
            n_observations = S[tr].shape[1]
            Z_hat[tr] = np.zeros((M,n_observations))
            
            # Perform first measurement update.
            if self.steady_state:
                Z_hat[tr][:,0] = z0 + K_inf @ (S[tr][:,0] - C @ z0)
            else:
                P = P0
                K = P @ C.T @ pinv(C @ P @ C.T + R)
                Z_hat[tr][:,0] = z0 + K @ (S[tr][:,0] - C @ z0)
                P -= K @ C @ P
            
            # Estimate iteratively.
            for t in range(n_observations-1):
                
                # Perform time update.
                Z_hat[tr][:,t+1] = A @ Z_hat[tr][:,t]
                if not self.steady_state:
                    P = A @ P @ A.T + Q
                
                # Perform measurement update.
                if self.steady_state:
                    Z_hat[tr][:,t+1] += K_inf @ (S[tr][:,t+1] - C @ Z_hat[tr][:,t+1])
                else:
                    K = P @ C.T @ pinv(C @ P @ C.T + R)
                    Z_hat[tr][:,t+1] += K @ (S[tr][:,t+1] - C @ Z_hat[tr][:,t+1])
                    P -= K @ C @ P

        # Remove acceleration and constant offset from estimates.
        Z_hat = [Z[:4,:] for Z in Z_hat]

        # Prepend with NaNs to account for lag in estimates.
        if bin_lag > 0:
            Z_hat = [np.hstack((np.full((Z.shape[0],bin_lag), np.nan), Z)) for Z in Z_hat]

        # Return estimate to original time scale.
        Z_hat = [zero_order_hold(Z,Delta) for Z in Z_hat]
        Z_hat = [np.hstack((np.full((Z.shape[0],Delta-1), np.nan), Z)) for Z in Z_hat]
        Z_hat = [z[:,:t] for z,t in zip(Z_hat, T)]
                
        return Z_hat

########################## FEEDFORWARD NEURAL NETWORK ##########################

class FeedforwardNetwork(object):

    """
    Class for the feedforward neural network decoder

    Hyperparameters
    ---------------
    Delta: number of time points to pool into a time bin
    
    tau_prime: number of previous time bins (in addition to the current bin) to use for decoding
    
    num_units: number of units per hidden layer

    num_layers: number of hidden layers

    frac_dropout: unit dropout rate

    num_epochs: number of training epochs
    
    """

    def __init__(self,HyperParams):
        self.Delta = HyperParams['Delta']
        self.tau_prime = HyperParams['tau_prime']
        self.num_units = HyperParams['num_units']
        self.num_layers = HyperParams['num_layers']
        self.frac_dropout = HyperParams['frac_dropout']
        self.num_epochs = HyperParams['num_epochs']

    def fit(self, S, Z):

        """
        Train feedforward neural network.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Z: list of M x T numpy arrays, each of which contains behavioral data for M behavioral variables over T times
            
        Parameters
        ----------
        net: Keras sequential neural network model
            
        """

        # Unpack attributes.
        Delta = self.Delta
        tau_prime = self.tau_prime
        num_units = self.num_units
        num_layers = self.num_layers
        frac_dropout = self.frac_dropout
        num_epochs = self.num_epochs

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]

        # Reformat observations to include recent history.
        X = [append_history(s, tau_prime) for s in S]

        # Downsample kinematics to bin width.
        Z = [z[:,Delta-1::Delta] for z in Z]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        X = [x[:,tau_prime:,:] for x in X]
        Z = [z[:,tau_prime:] for z in Z]

        # Concatenate X and Z across trials (in time bin dimension) and rearrange dimensions.
        X = np.moveaxis(np.concatenate(X,axis=1), [0, 1, 2], [1, 0, 2])
        Z = np.concatenate(Z, axis=1).T

        # Z-score inputs.
        X_mu = np.mean(X, axis=0)
        X_sigma = np.std(X, axis=0)
        X = (X - X_mu) / X_sigma
        self.X_mu = X_mu
        self.X_sigma = X_sigma

        # Zero-center outputs.
        Z_mu = np.mean(Z, axis=0)
        Z = Z - Z_mu
        self.Z_mu = Z_mu

        # Construct feedforward network model.
        net = tf.keras.Sequential(name='Feedforward_Network')
        net.add(layers.Flatten())
        for layer in range(num_layers): # hidden layers
            net.add(layers.Dense(num_units, activation='relu'))
            if frac_dropout!=0: net.add(layers.Dropout(frac_dropout))
        net.add(layers.Dense(Z.shape[1], activation='linear')) # output layer
        net.compile(optimizer="Adam", loss="mse", metrics="mse")

        # Fit model.
        net.fit(X, Z, epochs=num_epochs)
        self.net = net

    def predict(self, S):

        """
        Predict behavior with trained feedforward neural network.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Outputs
        -------
        Z_hat: list of M x T numpy arrays, each of which contains decoded behavioral data for M behavioral variables over T times
   
        """

        # Unpack attributes.
        Delta = self.Delta
        tau_prime = self.tau_prime
        X_mu = self.X_mu
        X_sigma = self.X_sigma
        Z_mu = self.Z_mu
        net = self.net

        # Store each trial's length.
        T = [s.shape[1] for s in S]

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]

        # Store each trial's bin length.
        T_prime = [s.shape[1] for s in S]

        # Reformat observations to include recent history.
        X = [append_history(s, tau_prime) for s in S]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        X = [x[:,tau_prime:,:] for x in X]

        # Concatenate X across trials (in time bin dimension) and rearrange dimensions.
        X = np.moveaxis(np.concatenate(X,axis=1), [0, 1, 2], [1, 0, 2])

        # Z-score inputs.
        X = (X - X_mu) / X_sigma

        # Generate predictions.
        Z_hat = net.predict(X)

        # Add mean back to outputs.
        Z_hat += Z_mu

        # Split Z_hat back into trials and transpose kinematic arrays.
        Z_hat = array2list(Z_hat, np.array(T_prime)-tau_prime, axis=0)
        Z_hat = [Z.T for Z in Z_hat]

        # Add NaNs where predictions couldn't be made due to insufficient spiking history.
        Z_hat = [np.hstack((np.full((Z.shape[0],tau_prime), np.nan), Z)) for Z in Z_hat]

        # Return estimate to original time scale.
        Z_hat = [zero_order_hold(Z,Delta) for Z in Z_hat]
        Z_hat = [np.hstack((np.full((Z.shape[0],Delta-1), np.nan), Z)) for Z in Z_hat]
        Z_hat = [z[:,:t] for z,t in zip(Z_hat, T)]

        return Z_hat

######################### GATED RECURRENT UNIT NETWORK #########################

class GRU(object):

    """
    Class for the GRU decoder

    Hyperparameters
    ---------------
    Delta: number of time points to pool into a time bin
    
    tau_prime: number of previous time bins (in addition to the current bin) to use for decoding
    
    num_units: number of units in the GRU layer

    frac_dropout: unit dropout rate

    num_epochs: number of training epochs
    
    """

    def __init__(self,HyperParams):
        self.Delta = HyperParams['Delta']
        self.tau_prime = HyperParams['tau_prime']
        self.num_units = HyperParams['num_units']
        self.frac_dropout = HyperParams['frac_dropout']
        self.num_epochs = HyperParams['num_epochs']

    def fit(self, S, Z):

        """
        Train GRU.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Z: list of M x T numpy arrays, each of which contains behavioral data for M behavioral variables over T times
            
        Parameters
        ----------
        net: Keras sequential neural network model
            
        """

        # Unpack attributes.
        Delta = self.Delta
        tau_prime = self.tau_prime
        num_units = self.num_units
        frac_dropout = self.frac_dropout
        num_epochs = self.num_epochs

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]

        # Reformat observations to include recent history.
        X = [append_history(s, tau_prime) for s in S]

        # Downsample kinematics to bin width.
        Z = [z[:,Delta-1::Delta] for z in Z]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        X = [x[:,tau_prime:,:] for x in X]
        Z = [z[:,tau_prime:] for z in Z]

        # Concatenate X and Z across trials (in time bin dimension) and rearrange dimensions.
        X = np.moveaxis(np.concatenate(X,axis=1), [0, 1, 2], [2, 0, 1])
        Z = np.concatenate(Z, axis=1).T

        # Z-score inputs.
        X_mu = np.mean(X, axis=0)
        X_sigma = np.std(X, axis=0)
        X = (X - X_mu) / X_sigma
        self.X_mu = X_mu
        self.X_sigma = X_sigma

        # Zero-center outputs.
        Z_mu = np.mean(Z, axis=0)
        Z = Z - Z_mu
        self.Z_mu = Z_mu

        # Construct GRU network model.
        net = tf.keras.Sequential(name='GRU_Network')
        net.add(layers.GRU(num_units, dropout=frac_dropout, recurrent_dropout=frac_dropout))
        if frac_dropout!=0: net.add(layers.Dropout(frac_dropout))
        net.add(layers.Dense(Z.shape[1], activation='linear'))
        net.compile(optimizer="RMSprop", loss="mse", metrics="mse")

        # Fit model.
        net.fit(X, Z, epochs=num_epochs)
        self.net = net

    def predict(self, S):

        """
        Predict behavior with trained GRU.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Outputs
        -------
        Z_hat: list of M x T numpy arrays, each of which contains decoded behavioral data for M behavioral variables over T times
   
        """

        # Unpack attributes.
        Delta = self.Delta
        tau_prime = self.tau_prime
        X_mu = self.X_mu
        X_sigma = self.X_sigma
        Z_mu = self.Z_mu
        net = self.net

        # Store each trial's length.
        T = [s.shape[1] for s in S]

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]

        # Store each trial's bin length.
        T_prime = [s.shape[1] for s in S]

        # Reformat observations to include recent history.
        X = [append_history(s, tau_prime) for s in S]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        X = [x[:,tau_prime:,:] for x in X]

        # Concatenate X across trials (in time bin dimension) and rearrange dimensions.
        X = np.moveaxis(np.concatenate(X,axis=1), [0, 1, 2], [2, 0, 1])

        # Z-score inputs.
        X = (X - X_mu) / X_sigma

        # Generate predictions.
        Z_hat = net.predict(X)

        # Add mean back to outputs.
        Z_hat += Z_mu

        # Split Z_hat back into trials and transpose kinematic arrays.
        Z_hat = array2list(Z_hat, np.array(T_prime)-tau_prime, axis=0)
        Z_hat = [Z.T for Z in Z_hat]

        # Add NaNs where predictions couldn't be made due to insufficient spiking history.
        Z_hat = [np.hstack((np.full((Z.shape[0],tau_prime), np.nan), Z)) for Z in Z_hat]

        # Return estimate to original time scale.
        Z_hat = [zero_order_hold(Z,Delta) for Z in Z_hat]
        Z_hat = [np.hstack((np.full((Z.shape[0],Delta-1), np.nan), Z)) for Z in Z_hat]
        Z_hat = [z[:,:t] for z,t in zip(Z_hat, T)]

        return Z_hat
