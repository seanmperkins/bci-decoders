# Brain-Computer Interface Decoders
Brain-computer interfaces (BCIs) encompass a class of technologies that allow the brain to interact with external devices. BCIs can utilize a variety of neural recording modalities and may be focused on control (e.g., moving a prosthetic arm by decoding neural activity) or feedback (e.g., restoring lost sense of touch by injecting current into the brain). This code package focuses on a specific niche of BCIs: decoding intended movement based on invasive neural recordings that include individual action potentials ('spikes'). The goal is to develop statistical techniques ('decoders') for accurately and causally predicting movement variables (e.g., x- and y-velocity of the arm) based solely on recorded spike times from a collection of neurons.

This repository benchmarks four common neural decode algorithms (Kalman Filter, Wiener Filter, Feedforward Neural Network, and GRU) for offline decoding on three data sets involving motion of a monkey's arm (two data sets with recordings from the motor cortex, and one data set with recordings from the somatosensory cortex). Detailed information on the data sets can be found on a separate [data README](data/) page. My hope is that this package will make it easier for others to benchmark their own algorithms against the decoders presented here. The repository can also serve as a resource for new researchers familiarizing themselves with neural decoding. 

# Setup

To install dependencies, create a virtual environment with Python 3.7.10 and run:

```
pip install -r requirements.txt
```

# Usage
The notebook `run_decoders.ipynb` contains commented example code for training each neural decoder, testing on a held-out test set, and visualizing the results of each run. A few settings can be configured directly in the notebook, but most settings, hyperparameters, and details related to hyperparameter optimization can be found in data-set-specific config files. More information on how to use the config files can be found on a separate [config README](config/) page.

### Note on Hyperparameter Optimization
Performance can be improved by selecting the right set of hyperparameters for a given method. There is an option in `run_decoders.ipynb` to optimize hyperparameters using Bayesian optimization. Be aware that this can add considerable time to the training procedure. This will be particularly noticeable for larger data sets (like MC_Maze) and more computationally intensive decode algorithms (like Feedforward Neural Network or GRU). For computationally intensive runs, consider using a machine with access to GPUs.

# References
### Data
The data sets provided here are processed versions of data sets released by the [Neural Latents Benchmark](https://neurallatents.github.io/). Journal articles associated with the original data sets are listed below and more detailed information can be found on the [data README](data/) page.
* [Chowdhury et al. 2020](https://elifesciences.org/articles/48198) (Area2_Bump)
* [Churchland et al. 2010](https://pubmed.ncbi.nlm.nih.gov/21040842/) (MC_Maze)
* [Makin et al. 2018](https://iopscience.iop.org/article/10.1088/1741-2552/aa9e95) (MC_RTT)

### Decoders
All four decode algorithms implemented here are common techniques that aren't specific to neuroscientific data. However, the references below contain descriptions of these techniques specifically for neural decoding.
* [Wu et al. 2003](https://www.dam.brown.edu/people/elie/papers/Wu%20et%20al%20NIPS%2003.pdf), [Gilja et al. 2012](https://pubmed.ncbi.nlm.nih.gov/23160043/) (Kalman Filter)
* [Carmena et al. 2003](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0000042) (Wiener Filter)
* [Glaser et al. 2020](https://www.eneuro.org/content/7/4/ENEURO.0506-19.2020) (all four decoders)

### Hyperparameter Optimization
We optimize hyperparameters using Bayesian global optimization with Gaussian processes. We use an existing [code package](https://github.com/fmfn/BayesianOptimization) associated with the conference paper [Snoek et al. 2012](https://proceedings.neurips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf).
