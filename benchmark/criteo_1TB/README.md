# Criteo 1TB Dataset

This readme provides information about the Criteo dataset, the preprocessing steps used and additional useful information.


## <ins>Dataset Information</ins>

The training dataset consists of a portion of Criteo’s traffic over a period of 24 days.
Each row corresponds to a display ad served by Criteo and the first column is indicates whether this ad has been clicked or not.
The positive (clicked, label = 1) and negatives (non-clicked, label = 0) examples have both been subsampled (but at different rates) in order to reduce the dataset size.

There are 13 features represented using integer values (mostly count features) and 26 categorical features.
The values of the categorical features have been hashed onto 32 bits for anonymization purposes.
The semantic of these features is undisclosed.
Some features may have missing values.

The rows are chronologically ordered.

It can be dowloaded from: https://labs.criteo.com/2013/12/download-terabyte-click-logs-2/


## <ins>Preprocessing for Modyn:</ins>

The preprocessing is done for a couple of reasons.
Firstly, there are a lot of null values present and also some categorical values have a large range with small frequency (eg 5 million+ unique values with a lot of them appearing only once or twice).
For efficiency, we only want to learn those features with a high frequency (in this case 15 or more) and hence we filter out values with a low frequency into one single value.
For example if a category 'c1' had the values 'unique1', 'unique2' and 'unique3' appearing only once, we would like to convert them to the same default value, say '0' such that the total number of unique values in that category reduces.
In this way, if there were 100 different values in the category, but only 10 of them appeared more than 15 times, we would need only 11 values to represent the data in the column (1 for the default and 10 for the 10 high frequency values) rather than 100.
Removing out the low frequency values by mapping them all to the same default value can help speedup the model a lot at the cost of a little accuracy.
Also we want to convert the format into something that takes less space and is more effeciently read.
The final preprocessing converts all the input files into binary files while re embedding the categorical values to filter out the low frequency values.


### Preprocessing Steps
We build upon the preprocessing scripts found in the [NVIDIA Deep Learning Example repository](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Recommendation/DLRM/README.md).
The preprocessing in the NVIDIA repository is setup to embed the categorical features, as well as split the 24 days of data into a train, validation and test set.
It uses the first 23 days as the train set and divides the data of the last day equally into the validation and the test sets.

Since we want a more incremental approach (to simulate data incoming daily), we wanted to split the data set into more modular parts.
The script was edited to split the data into slightly different batches.
Instead of train, test and validaiton, the data was processed and split into individual daily folders, example one for day 0, one for day 1, and so on.

The steps to run the preprocessing as well as the patch to apply to the above scripts are present in the '/preprocessing/ folder.


### Output
The preprocessed data is finally output as a set of binary files per day. For each day, there is an individual folder containing binary files. Each binary file contains a subset of the preprocessed samples.
