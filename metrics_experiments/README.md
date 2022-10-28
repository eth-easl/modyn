# Code Structure

This code is designed to be as modular and configurable as possible. The three main modular pieces are the dataset, model, and trainer. Not all (dataset, model, trainer) tuples are compatible with each other! For example, the current trainers are set up to handle classification tasks. A good starting point to see what the configurable options are for each of these is the builder.py file. 

Run the code very simply: ``python train.py {experiment_name}``. Optionally add ``--write`` to write out the results to the ``plots/`` folder. 

# Experimental Setup

## Regime

The regime we are considering is called "Class Incremental Learning" (CIL) in literature. In this setup, you present 

Blurry CIL refers to the case where we do allow for some instances from "minor classes", or those that aren't the focus of the current task, into our current dataset as well. So a BlurryM setup refers to the fact that M% of the current dataset is from the minor classes (evenly distributed). In the extremal case that M=0, we can equivalently call this the Disjoint CIL setup. We expect, of course, that higher M results in easier learning. 

Offline learning refers to the idea that we can keep the current task in memory. Essentially, we can see the whole current task and then choose which parts of it we want to keep. In online learning, we restrict ourselves to only be able to see samples once, unless we add it to our buffer. Online learning is clearly harder, and demands even more careful memory management techniques. 

## Metrics

The popular metrics reported for these tasks are Ax (Last Accuracy), Fx (Last Forgetting), and Ix (Intransigence), where x is the number of tasks (5 for SplitMNIST/SplitCIFAR and 10 for ImageNet). Last accuracy refers to the accuracy on the whole validation set after the entire training period. Last Forgetting averages the forgetting of all tasks, where forgetting is (max validation - ending validation) for the task. Finally, intransigence is the per-task accuracy gap to a model that was trained on all the classes (that is, not in the CIl setting). 


# Other approaches

## 
EWC

Rwalk

iCaRL

GDumb

BiC


# Experiments Overview

The naming scheme for experiments is (arbitrarily) as follows. For each experiment, create a name that contains the dataset name as well as an indication of what kind of sampling was used. For example, splitmnist_uniform. Then, you can also name the variant in the particular trial. For example, "control" or "reset_model". Then the config file should be named {experiment_name}-{trial_name}. For just trying things out, I use {dataset}1, {dataset}2, etc. 

Below are a list of experiments currently completed and the basic conclusion to be drawn. 

1. splitmnist_uniform
    1. control: Got accuracy of 92.17, in line with bilevel coresets paper. 
    2. reset_model: Got accuracy of 89.82, slightly higher than the bilevel coresets paper. 

2. splitmnist_highloss
    1. control: Got accuracy of 91.7