# Functional Map of the World dataset (from Wild-time)

In this directory, you find files necessary to run experiments on the FMoW dataset. The goal is to predict land use (62 classes)
given a satellite image . The dataset contains more than 100.000 samples collected from 2002 to 2017. Every picture is stored
separately (in png format) and the os timestamp is set accordingly.

This dataset is taken from wild-time and processed by adapting scripts from wild-time-data. You can find more details 
[here](https://github.com/huaxiuyao/Wild-Time)

## Data Generation

The script for downloading the data is adapted from `wild-time-data`. 
You can use the `data_generation.py` script to download the dataset. 
Use the `-h` flag to find out more.
