# Config Files
Each data set has a YAML config file containing settings, hyperparameters, and details related to hyperparameter optimization. The files are structured hierarchically with the top level indicating the decoder. Decoders are specified using the following syntax:

`kf`: Kalman Filter

`wf`: Wiener Filter

`ffn`: Feedforward Network

`gru`: Gated Recurrent Unit (GRU)

## Behavioral Variable Groups

Within each decoder's section, there is a field `var_groups` that determines which behavioral variables to decode and how to group them for decoding. `var_groups` is formatted like a Python list where each element specifies a group of behavioral variables that should be decoded together by a single decoder. If an element of the list is a string, then a single decoder is trained to decode the behavioral variable group indicated by that string. If an element of the list is itself a list of behavioral variable groups, then a single decoder will be trained to simultaneously output all of the behavioral variable groups in the nested list. For example, Area2_Bump has fields for hand position (`pos`), hand velocity (`vel`), and joint angles (`joint_ang`), among others. Below is example syntax for two different scenarios.

If you want to train two decoders, one for hand position and another for hand velocity...

`var_groups: ['pos','vel']`

If you want to train two decoders, one that outputs hand position *and* hand velocity, and a second that outputs joint angles...

`var_groups: [['pos','vel'],'joint_ang']`

Each data set has a different set of behavioral variables associated with it (see [data README](../data) for complete details). The Kalman filter is written to expect both 2D position and 2D velocity as inputs, so it should always be specified as...

`var_groups: [['pos','vel']]`

## General Hyperparameters

Within each decoder's section, the `general` field contains hyperparameters that won't change from one behavioral variable group to the next. These hyperparameters are set by the user and do not get optimized. These include...

`Delta`: Bin width (milliseconds).

`tau_prime`: Number of previous time bins to use for decoding (not including the current time bin).

The Kalman filter only operates on one time bin at a time and therefore doesn't use the `tau_prime` hyperparameter directly. However, `tau_prime` also determines the portion of each trial that can get evaluated. For example, if decoding causally with 200 ms of spiking history, an estimate cannot be rendered until at least 200 ms have elapsed in each trial. Thus, we still set `tau_prime` for the Kalman filter to match the spiking history used by other decoders when we wish to compare to other decoders. Despite not needing the full history, the code package will still exclude early trial estimates for the Kalman filter based on `tau_prime` to ensure all decoders are being evaluated over the same portion of each trial.

There is one additional general hyperparameter called `steady_state` that only gets set for the Kalman filter. It is a boolean that determines whether to use the steady-state form of the Kalman filter (True) or the standard Kalman filter (False).

## Hyperparameter Optimization
There are additional hyperparameters listed in the table below, specific to each decoder, that one may wish to optimize individually for each behavioral variable group.

| Decoder | Hyperparameter | Description |
| --- | --- | --- |
| Kalman Filter | `lag` | Value determining how many time samples (milliseconds) neural activity should be shifted relative to behavioral variables to reflect physiological lag between neural activity and behavior. Suppose `Delta = 20` and `lag = 0`. In this case, spikes from samples 1-20 would be binned and used to predict behavior for the 20th sample. If instead `lag = 100`, those same spikes would be used to predict behavior for the 120th sample.
| Wiener Filter | `lam` | L2 regularization term in the ridge regression to fit the filter weights. |
| Feedforward Network | `num_units` | Number of units per hidden layer. |
| Feedforward Network | `num_layers` | Number of hidden layers. |
| Feedforward Network | `frac_dropout` | Fraction of units from each hidden layer that should be dropped out during training. |
| Feedforward Network | `num_epochs` | Number of training epochs. |
| GRU | `num_units` | Number of units in the GRU layer. |
| GRU | `frac_dropout` | Fraction of units (both for the linear transformation of the inputs and the linear transformation of the recurrent state) that should be dropped out during training. |
| GRU | `num_epochs` | Number of training epochs. |

We utilize a pre-existing [Bayesian optimization package](https://github.com/fmfn/BayesianOptimization) to find hyperparameters that maximize prediction R<sup>2</sup> on held-out validation sets. The settings related to optimization are listed under the `opt` field. The primary settings are:

`init_points`: number of steps of random exploration of hyperparameter space to initially perform

`n_iter`: number of steps of Bayesian optimization of hyperparameters to subsequently perform

`kappa`: determines how much the acquisition function balances exploration vs. exploitation (higher values yield more relative exploration)

`val_frac`: fraction of training set to hold out as a validation set

Bounds for each hyperparameter are additionally set in the `opt` field in list format. For example, if optimizing `num_layers` for `ffn`, under `opt` we would write `num_layers: [1,10]` to indicate that the number of layers to consider should be somewhere between 1 and 10.

Note that optimizing hyperparameters is optional and will considerably increase the time it takes to run the decoders. The `optimize_flag` in `run_decoders.ipynb` is a boolean dictating whether to perform optimization. If true, it will use the procedure described in this section. If false, it will expect to be provided with values to use for the relevant hyperparameters. To specify values for this latter case, add additional fields (one level below the decoder field in the config file) with the name of each element in `var_group`. Inside those fields, fixed hyperparameters can be defined. If a `var_group` element is itself a list of behavioral variable groups, give the field a name that concatenates the elements of that list, separating them by hyphens.

For example, if we are writing the section for `wf` where `var_groups: [['pos','vel'],'joint_ang']`, we might write:

```
wf:
  var_groups: [['pos','vel'],'joint_ang']
  general:
    Delta: 20
    tau_prime: 11
  opt:
    lam: [0,2000]
    init_points: 10
    n_iter: 10
    kappa: 10
    val_frac: 0.2
  pos-vel:
    lam: 60
  joint_ang:
    lam: 0
```

