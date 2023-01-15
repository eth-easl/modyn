# Criteo 1TB Advertising Dataset

This readme provides information about the Criteo dataset, the preprocessing steps used and additional useful information.


## <ins>Criteo 1TB Dataset</ins>

The training dataset consists of a portion of Criteo’s traffic over a period
of 24 days. Each row corresponds to a display ad served by Criteo and the first
column is indicates whether this ad has been clicked or not.
The positive (clicked) and negatives (non-clicked) examples have both been
subsampled (but at different rates) in order to reduce the dataset size.

There are 13 features taking integer values (mostly count features) and 26
categorical features. The values of the categorical features have been hashed
onto 32 bits for anonymization purposes.
The semantic of these features is undisclosed. Some features may have missing values.

The rows are chronologically ordered.

It can be dowloaded from: https://labs.criteo.com/2013/12/download-terabyte-click-logs-2/




## <ins>Preprocessing for Modyn:</ins>

As mentioned the raw dataset has categorical features that take up a set of values. This needs to be embeded before being passed into the model, which is the major part of the preprocessing.

The preprocessing parses all the data and creates an embedding for each of the categorical values.


### Preprocessing Steps
For this purpose we have used the preprocessing scripts found in the NVIDIA Deep Learning Example repository - https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Recommendation/DLRM/README.md 

All the steps for preprocessing are found in that repository (specifically under the Quick Start Guide). It provides all the information in order to downlooad and pre process the dataset. 


### Changes
Note that the above steps is setup to convert the data into the a train, test and validation split (Days 0-22 are converted to train while day 23 is split 50-50 into a test and validation).

Since we want a more incremental approach, we wanted to split the data set into more modular parts. The script was edited to split the data into slightly different batches. Instead of train, test and validaiton, the data was processed and split into "day0-18", "day19", "day20", "day21", "day22" and "day23"

All detailed steps including the changes as well as a step by step process on how to setup a cloud cluster and run it are given in the "/preprocessing" folder.


### Output
The preprocessed data is finally output as a set of parquet files per "split". For each split, example day 23, there is a folder containing parquet files that hold the pre processed rows for all data for that day.




## <ins>Persistent Storage of the Data</ins>
Having processed the data, the processed data has been uploaded to Google Cloud Storage under the bucket "dds-criteo" in the dynamic-datasets-eth project.
This can be accessed either by the console, or via Google Storage API commands.

![Image](https://github.com/eth-easl/dynamic_datasets_dsl/benchmark/criteo_1TB/gcs_dds-criteo.png)

The corrsponding data can be found in "output/" where each directory there (eg day23/) contains the parquet files for the data for that particular data.

Additionally there are a set of files under the directiory "binary_dataset/" which is the same data as the parquet files, but processed as a binary file. We would not be using that. 
