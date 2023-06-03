# Yearbook dataset (from Wild-time)

In this directory, you find files necessary to run experiments on the yearbook dataset. The goal is to predict the sex given a 
yearbook picture. The dataset contains 37189 samples collected from 1930 to 2013. Since timestamps in Modyn are based on 
Unix Timestamps (so 0 is 1/1/1970) we have to remap the years to days. Precisely, the timestamp for pictures from 1930 is
1/1/1970, then 2/1/1970 for the ones taken in 1931 and so forth. 

This dataset is taken from wild-time and processed by adapting scripts from wild-time-data. You can find more details 
[here](https://github.com/huaxiuyao/Wild-Time)

## Data Generation
To run the downloading script you need to install the `gdown` library.

The script for downloading the data is adapted from `wild-time-data`. 
You can use the `data_generation.py` script to download the dataset. 
Use the `-h` flag to find out more.

A binary file for every year is produced. Labels are 4bits in size while records are 4100.
You can use the following config to parse the file ```{
    "record_size": 4100,
    "label_size": 4,
    "byteorder": "big",
}```