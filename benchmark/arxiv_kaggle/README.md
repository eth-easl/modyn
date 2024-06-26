# Arxiv Dataset (sourced from Kaggle)

In this directory, you can find the files necessary to run experiments using the Arxiv dataset sourced from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data).

## Dataset

The goal is to predict the paper category (172 classes) given the paper title.
The dataset contains more than 2 million samples collected from 1986 to 2024.
Titles belonging to the same year are grouped into the same CSV file and stored together.

## Data Download

Please download the dataset from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data) and place the `.zip` file it in your desired location.

## Data Generation

```bash
python -m benchmark.arxiv_kaggle.data_generation
    .debug.log/datasets/arxiv_kaggle  # destination folder
    .debug.log/datasets/arxiv_kaggle/_raw/archive.zip  # manual download from Kaggle
    --resolution day
    --test-split
    --dummy-period
```

The data generation script relies on the `raw_data` argument which specifies the path to the downloaded `.zip` file. The script will extract the zip to a `arxiv-metadata-oai-snapshot.json` file and generate the necessary dataset files for the dataset including some light preprocessing.

### Preprocessing

#### Binning into years

Depending on the desired resolution (yearly batches, monthly batches, daily, ...) the data can be binned into different time intervals. Samples in the same time resolution bin will end up in the same data file.

You can specify the time resolution using the `--resolution` argument.

#### Cleaning

The dataset contains more features than we use. We actually only use the title (input), the category (target) and the first version timestamp.

- title: The title of the paper; we strip newlines and reduce whitespace to single spaces
- category: The category of the paper; we only use the primary category; there are 
- first_version_timestamp: we extract the first release date of the first version of the paper. Every revision of the paper has an own timestamp, however, we only use the first one.
