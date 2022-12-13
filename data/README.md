# Data Format
This repository contains three data sets: Area2_Bump, MC_Maze, and MC_RTT. These data sets were collected by academic laboratories and released by the [Neural Latents Benchmark](https://neurallatents.github.io/) in the [Neurodata Without Borders](https://www.nwb.org/) format. For convenience, they have been preprocessed into .pickle files that are smaller, trialized, and only contain data relevant to this code package.

Each .pickle file contains a dictionary `Data` that can be loaded via:
```
import pickle
dataset = 'area2_bump'
with open('data/' + dataset + '.pickle','rb') as f:
    Data = pickle.load(f)
```

`Data` is a dictionary with several keys (e.g., `spikes`, `condition`, `pos`), each of which contains trialized data. `spikes` is a list whose length matches the number of trials in the data set. Each element of the list contains an N x T numpy array of spike counts binned at 1 ms resolution where N is the number of neurons and T is the number of 1 ms samples in the trial. Behavioral variable groups like `pos` are similarly stored as a list of M x T numpy arrays where M is the number of components that describe that behavioral variable group (e.g, M = 2 for position when there is an x- and y-component). Area2_Bump and MC_Maze contain a `condition` key that is an numpy array of condition IDs, one per trial.

## Area2_Bump
This data set contains neural recordings from area 2 of the somatosensory cortex while a monkey uses a manipulandum to make center-out-reaches to one of eight targets. Some trials involve a volitional center-out-reach ('active' trials) while in some trials the monkey is perturbed out to the target ('passive' trials). Thus, right after movement onset the 'active' trials involve predictable sensory feedback and the 'passive' trials involve unpredictable sensory feedback. There are 364 trials in the data set provided, each beginning 339 ms before movement onset and ending 500 ms after movement onset.

### Conditions
There are 8 reach directions and 2 trial types ('active' and 'passive') for a total of 16 conditions. The `condition` field contains a condition number from 1 to 16. Odd condition numbers correspond to 'active' reaches to the 8 targets. Even condition numbers correspond to 'passive' reaches to the 8 targets. Condition numbers 1 & 2 correspond to a reach to the 0-degree target, 3 & 4 correspond to a 45 degree target, etc.

### Behavioral Variable Groups
`pos`: x- and y-position of the monkey's hand (cm)

`vel`: x- and y-velocity of the monkey's hand (cm/s)

`force`: forces and torques applied to the manipulandum (1st, 3rd, and 5th components correspond to x-, y-, and z-forces; 2nd, 4th, and 6th components correspond to x-, y-, and z-torques)

`joint_ang`: angles of 7 of the monkey's arm joints (degrees)

`joint_vel`: velocity of 7 of the monkey's arm joints (degrees/s)

`muscle_len`: length of 39 of the monkey's arm muscles (m)

`muscle_vel`: velocity of 39 of the monkey's arm muscles (m/s)

### Attribution
This data was provided to the Neural Latents Benchmark by Raeed Chowdhury and Lee Miller at Northwestern University. The full data set is available on [DANDI](https://dandiarchive.org/dandiset/000127) and more information about the data can be found in the journal article [Chowdhury et al. 2020](https://elifesciences.org/articles/48198).

## MC_Maze
This data set contains neural recordings from M1 and PMd while a monkey makes delayed center-out straight and curved reaches. The monkey was presented with a virtual maze with targets to reach toward. Virtual barriers were presented that the monkey had to avoid colliding with when reaching, which prompted curved reaches to avoid the barriers. On some trials three targets were presented, but only one target was reachable via the maze. There are 2,295 trials in the data set provided, each beginning 549 ms before movement onset and ending 450 ms after movement onset.

### Conditions
There are 108 conditions in this data set: 36 maze configurations with 3 variants per maze. The `condition` field contains a condition number from 1 to 108. Condition numbers 1-3 correspond to one maze configuration, condition numbers 4-6 correspond to a second maze configuration, etc. Within each maze configuration, the first condition number corresponds to a reach to a target with no barrier (straight reach), the second condition number corresponds to reaching to a target with a barrier (curved reach), and the third condition number corresponds to reaching to a target with a barrier (curved reach) in the presence of two unreachable distractor targets.

### Behavioral Variable Groups

`pos`: x- and y-position of the monkey's hand (mm)

`vel`: x- and y-velocity of the monkey's hand (mm/s)

### Attribution
This data was provided to the Neural Latents Benchmark by Krishna Shenoy, Mark Churchland, and Matt Kaufman at Stanford University. The full data set is available on [DANDI](https://dandiarchive.org/dandiset/000128) and more information about the data can be found in the journal article [Churchland et al. 2010](https://pubmed.ncbi.nlm.nih.gov/21040842/).

## MC_RTT
This data set contains neural recordings from M1 while a monkey makes self-paced reaches to random targets in an 8x8 grid. There is no condition structure in the data set. For the Neural Latents Benchmark, the session was broken up into 600 ms trials, with no particular alignment relative to movement on a given trial. The 1076 trials provided here have an additional 279 ms of data prepended to each trial. This is provided to allow causal decoders access to a reasonably long history such that causal estimates can generate predictions over the 600 ms period. For example, if decoding based on a trailing history of 280 ms, one could not generate a decode for the first 279 ms of each trial, but all subsequent samples in the trial would receive a prediction. 

### Behavioral Variable Groups

`pos`: x- and y-position of the monkey's finger (mm)

`vel`: x- and y-velocity of the monkey's finger (mm/s)

### Attribution
This data was provided to the Neural Latents Benchmark by Joseph O'Doherty and Philip Sabes at the University of California San Francisco. The full data set is available on [DANDI](https://dandiarchive.org/dandiset/000129) and more information about the data can be found in the journal article [Makin et al. 2018](https://iopscience.iop.org/article/10.1088/1741-2552/aa9e95).

# Data Storage
The .pickle files exceed GitHub's 100 MB file size limit. As a workaround, pointers to the data files are tracked in this repository using Git Large File Storage.
