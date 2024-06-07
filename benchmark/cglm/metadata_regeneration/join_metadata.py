import os

import pandas as pd


def read_second_csvs(directory):
    dfs = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            dfs.append(pd.read_csv(file_path, dtype={'upload_date': 'int'}))

    return pd.concat(dfs, ignore_index=True).drop("filename", axis=1).rename(columns={'fileid': 'id'})

# Directory containing the timestamp csv files
second_csv_directory = '/scratch/maximilian.boether/scrape/outputs'

full_source = pd.read_csv('/scratch/maximilian.boether/scrape/train_attribution.csv')
second_csv_file_ids = read_second_csvs(second_csv_directory)

# Attach timestamp to each file
timestamped_df = pd.merge(full_source, second_csv_file_ids, on='id', how='inner')

# Attach label to each file
landmarks_df = pd.read_csv('/scratch/maximilian.boether/scrape/train.csv').drop("url", axis=1)
labeled_df = pd.merge(timestamped_df, landmarks_df, on='id', how='inner')

# Attach clean marker
clean_df = pd.read_csv('/scratch/maximilian.boether/scrape/train_clean.csv')
clean_df['images'] = clean_df['images'].str.split()
landmark_dict = dict(zip(clean_df['landmark_id'], clean_df['images']))

def is_clean(row):
    file_id = row['id']
    landmark_id = row['landmark_id']
    if landmark_id in landmark_dict:
        return file_id in landmark_dict[landmark_id]
    return False

labeled_df['clean'] = labeled_df.apply(is_clean, axis=1)

hierarchy_df = pd.read_csv('/scratch/maximilian.boether/scrape/index_label_to_hierarchical.csv').drop('category', axis=1)
supercategory_to_label = {supercategory: label for label, supercategory in enumerate(hierarchy_df['supercategory'].unique())}
hierarchical_label_to_label = {hierarchical_label: label for label, hierarchical_label in enumerate(hierarchy_df['hierarchical_label'].unique())}
hierarchy_df['supercategory_label'] = hierarchy_df['supercategory'].map(supercategory_to_label)
hierarchy_df['hierarchical_label_label'] = hierarchy_df['hierarchical_label'].map(hierarchical_label_to_label)

final_df = pd.merge(labeled_df, hierarchy_df, on='landmark_id', how='inner')

final_df.to_csv("cglm_labels_timestamps_clean.csv", index=False)

