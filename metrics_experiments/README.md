## Code Structure

This code is designed to be as modular and configurable as possible. The three main modular pieces are the dataset, model, and trainer. Not all (dataset, model, trainer) tuples are compatible with each other! For example, the current trainers are set up to handle classification tasks. A good starting point to see what the configurable options are for each of these is the builder.py file. 

Run the code very simply: ``python train.py {experiment_name}``. Optionally add ``--write`` to write out the results to the ``plots/`` folder. 

## Experiments Overview

The naming scheme for experiments is (arbitrarily) as follows. For each experiment, create a name that contains the dataset name as well as an indication of what kind of sampling was used. For example, splitmnist_uniform. Then, you can also name the variant in the particular trial. For example, "control" or "reset_model". Then the config file should be named {experiment_name}-{trial_name}. For just trying things out, I use {dataset}1, {dataset}2, etc. 

Below are a list of experiments currently completed and the basic conclusion to be drawn. 

1. splitmnist_uniform
    1. control: Got accuracy of 92.17, in line with bilevel coresets paper. 
    2. reset_model: Got accuracy of 89.82, slightly higher than the bilevel coresets paper. 

2. splitmnist_highloss
    1. control: Got accuracy of 91.7