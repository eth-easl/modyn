# Wild-time datasets

In this directory, you can find the files necessary to run experiments using benchmarks from Wild-time. 
There are 4 available datasets: **arxiv**, **huffpost**, **FMoW** and **yearbook**. 
You can find more details in the [wild time repo](https://github.com/huaxiuyao/Wild-Time)

## Data Generation
To run the downloading script you need to install the `gdown` library and, just for FMoW, also the `wilds` library. 

The downloading scripts are adapted from `wild-time-data`. 
There is a `data_generation_[benchmark].py` script for each available dataset. 
Use the `-h` flag to find out more.


## Datasets description

### Yearbook
The goal is to predict the sex given a yearbook picture.
The dataset contains 37189 samples collected from 1930 to 2013. 
Since timestamps in Modyn are based on Unix Timestamps (so 0 is 1/1/1970) we have to remap the years to days. 
Precisely, the timestamp for pictures from 1930 is 1/1/1970, then 2/1/1970 for the ones taken in 1931 and so forth. 
Samples are saved using BinaryFileWrapper by grouping all samples of the same year in one file.

### FMoW
The goal is to predict land use for example, _park_, _port_, _police station_ and _swimming pool_, given a satellite image.
Due to human activity, satellite imagery changes over time, requiring models that are robust to temporal distribution shifts.
The dataset contains more than 100.000 samples collected from 2002 to 2017. 
Every picture is stored separately (in png format and loaded using SingleSampleFileWrapper) and the os timestamp is set accordingly.

### HuffPost
The goal is to predict the tag of news given headlines. 
The dataset contains more than 60k samples collected from 2012 to 2018. 
Titles belonging to the same year are grouped into the same CSV file and stored together. 
Each year is mapped to a year starting from 1/1/1970.

### Arxiv
The goal is to predict the paper category (55 classes) given the paper title. 
The dataset contains more than 2 million samples collected from 2002 to 2017. 
Titles belonging to the same year are grouped into the same CSV file and stored together. 
Each year is mapped to a year starting from 1/1/1970.

## DOWNLOAD UTILS license
Some code relies on the [Wild-Time-Data repository](https://github.com/wistuba/Wild-Time-Data). 
The copyright for that code lies at the author of the wild time data, wistuba. 
Please find a full copy of the license [here](https://raw.githubusercontent.com/wistuba/Wild-Time-Data/main/LICENSE)


## DATASET licenses
We list the licenses for each Wild-Time dataset below:

- Yearbook: MIT License
- FMoW: [The Functional Map of the World Challenge Public License](https://raw.githubusercontent.com/fMoW/dataset/master/LICENSE)
- Huffpost: CC0: Public Domain
- arXiv: CC0: Public Domain