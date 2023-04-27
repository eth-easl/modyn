# WIP: ArXiv dataset (from Wild-time)

### This benchmark is not ready to be used.
Downloading the data works but right now it is stored inefficiently using a SingleSampleFileWrapper.
In this way, there are more than 1 million small files which are slowly loaded into the DB.
We need an efficient FileWrapper to handle text based inputs.

## Description

In this directory, you find files necessary to run experiments on the ArXiv dataset. The goal is to predict paper category (55 classes)
given paper title. The dataset contains more than 2 million samples collected from 2002 to 2017. Every title is stored
separately and the os timestamp is set accordingly.

This dataset is taken from wild-time and processed by adapting scripts from wild-time-data. You can find more details 
[here](https://github.com/huaxiuyao/Wild-Time)

## Data Generation
To run the downloading script you need to install the `gdown` library.


The script for downloading the data is adapted from `wild-time-data`. 
You can use the `data_generation.py` script to download the dataset. 
Use the `-h` flag to find out more.
