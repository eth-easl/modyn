# CLOC Data

In this directory, you can find the files necessary to run experiments with the CGLM dataset. 
This dataset contains pictures of landmarks (e.g., the Montblanc) over time.
The dataset was introduced [in this paper](https://drimpossible.github.io/documents/ACM.pdf).
However, their host is down and we have to regenerate the dataset.
It makes use of the [Google Landmarks V2 dataset](https://github.com/cvdfoundation/google-landmark).

## Generating the dataset for Modyn

As a first step, you will need to download the Google Landmarks v2 Images.
You can check out [their repository](https://github.com/cvdfoundation/google-landmark) for instructions, or try this snippet:

```bash
wget -c https://raw.githubusercontent.com/cvdfoundation/google-landmark/master/download-dataset.sh
mkdir cglm && cd cglm
bash ../download-dataset.sh train 499
```

Next, we need the image metadata.
We have pregenerated and host the metadata on Github for convenience.
If you want to check out how we generate this metadata, check out the next file.

```bash
wget https://github.com/eth-easl/cglm-metadata/raw/main/cglm_labels_timestamps_clean.csv
```

Next, you can use the `data_generation.py` script in this directory to generate the dataset.
Note that there are different versions of CGLM you can generate:
- You can filter out classes that have less than `n` samples using the `--min_samples_per_class` flag. This defaults to 25.
- If you supply the `--clean` flag, then the [cleaned version](https://arxiv.org/abs/2003.11211) of the landmark dataset is used. This dataset has been cleaned to contain more consistent images of each landmark, but is a lot smaller.
- Using `--labeltype`, you can control which label we use. By default, we use the (remapped) landmark id. However, the team behind the landmark datasets supplies two classes of hierarchical labels (supercategory and hierarchical) that can alternatively be used, to make the classification problem smaller.

The script will tell you at the end how many classes your dataset contains.
Note that the size of the dataset is not consistent with what is reported in the initial paper on CGLM, but since their code on metadata processing is not open source, we cannot investigate the difference here.

## Regenerate the metadata
This paragraph is only relevant if you are interested in how we generated the `cglm_labels_timestamps_clean.csv` file.
For this, relevant scripts are supplied in the `metadata_regeneration` subdirectory.
First, we need to scrape the metadata from the commonsapi.
Note that you might consider hosting this API yourself (https://bitbucket.org/magnusmanske/magnus-toolserver/src/master/public_html/commonsapi.php) instead of using the public host that is in the script (`scrape.py`).
We thank the authors of the CGLM for the [initial version of the scraper](https://github.com/drimpossible/ACM/blob/main/scripts/cglm_scrape.py), which we extended to parallelize across machines, show the progress correctly, and be more robust.
Note that scraping the metadata might take several days here.

Next we need to parse the scraped XML data using the `xml_to_csv.py` script. 
Using both XML parsing and regex, this script tries to extract the upload date from the downloaded metadata, and generates a csv containing the timestamp for each file ID.

Next, we need to accumulate all those csvs on a single machine (in case you decided to parallelize the scraping and parsing across machines).
Furthermore, we need to download the additional metadata about what samples belong to the cleaned dataset, and the hierarchical labels to our directory.

```bash
wget https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv
wget https://s3.amazonaws.com/google-landmark/metadata/train.csv
wget https://s3.amazonaws.com/google-landmark/metadata/train_label_to_hierarchical.csv
```

Last, we join all this information using the `join_metadata.py` script, generating the final csv that you can also download from Github as outlined above.

## License

The Google Landmarks Dataset v2 annotations comes with a CC BY 4.0 license.
Check out their [repository](https://github.com/cvdfoundation/google-landmark) for more information.
The adjusted code used to scrape the metadata from wikipedia comes from the repository of the paper [Online Continual Learning Without the Storage Constraint
](https://github.com/drimpossible/ACM).
It is served under the following MIT license:

### CGLM Scrape License

MIT License

Copyright (c) 2022 Ameya Prabhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.