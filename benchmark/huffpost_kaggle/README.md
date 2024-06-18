# Huffpost Dataset (sourced from Kaggle)

In this directory, you can find the files necessary to run experiments using the Huffpost dataset sourced from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset).

## Dataset

The goal is to predict the tag of news given headlines.
The dataset contains more than 60k samples collected from 2012 to 2018.
Titles belonging to the same year are grouped into the same CSV file and stored together.
Each year is mapped to a year starting from 1/1/1970.
There is a total of 42 categories/classes.

> Note: The wild-time variant of the huffpost dataset has only 11 classes. This is due to the fact that
> the wild-time variant only considers tags that are present in all years.
> We argue that the introduction of new tags is also a relevant aspect of data drift and should be a factor in
> the evaluation of our pipelines. Stripping all data for a recently introduced topic/tag does not reflect reality.

## Data Download

Please download the dataset from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset) and place the `.zip` file it in your desired location.

## Data Generation

```bash
python -m benchmark.Huffpost_kaggle.data_generation
    .debug.log/datasets/Huffpost_kaggle  # destination folder
    .debug.log/datasets/Huffpost_kaggle/_raw/archive.zip  # manual download from Kaggle
    --resolution day
    --test-split
    --dummy-period
```

The data generation script relies on the `raw_data` argument which specifies the path to the downloaded `.zip` file. The script will extract the zip to a `Huffpost-metadata-oai-snapshot.json` file and generate the necessary dataset files for the dataset including some light preprocessing.

### Preprocessing

#### Binning into years

Depending on the desired resolution (yearly batches, monthly batches, daily, ...) the data can be binned into different time intervals. Samples in the same time resolution bin will end up in the same data file.

You can specify the time resolution using the `--resolution` argument.

#### Cleaning

The dataset contains more features than we use. We actually only use the title (input), the category (target) and the first version timestamp.

- title: The title of the article; we strip newlines and reduce whitespace to single spaces
- category: The category of the article encoded with integer codes
- date: release date of the article
