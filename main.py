import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import fnmatch
import os
import torch

from trainer import Trainer
from config import *



def get_files( path, extention):
        files = []
        for root, dirNames, fileNames in os.walk(path):
            for fileName in fnmatch.filter(fileNames, '*' + extention):
                files.append(os.path.join(root, fileName))
        return files


def load_data( dataset_path, data_info, extention='.mp3'):
    if not os.path.isdir(dataset_path):
        print('Dataset path does not exist ')
        quit()
    else:
        # Retrieve all files in chosen path with the specific extension
        all_data_paths = get_files(dataset_path, extention)
        # Create a dictionary where keys are unique parts (file names) and values are paths
        path_dict = {os.path.basename(path): path for path in all_data_paths}

        paths, texts, speaker_ids = [], [], []
        for index, row in tqdm(data_info.iterrows()):
            # The data_table's path actually is name of file, not its real path
            paths.append(path_dict[row.path])
            texts.append(row.sentence)
            speaker_ids.append(row.client_id)

        if len(paths) == 0:
            print('There is no sample in dataset')
            quit()
        else:
            return pd.DataFrame({'path': paths, 'text':texts, 'speaker_id':speaker_ids})


def dataset_report(train_data, test_data, speaker_counts, speaker_counts_filtered):
    # Calculate the total duration in hours, minutes, and seconds
    hours, minutes, seconds = get_duration(4 * len(train_data))
    print('-'*50)
    print(f'Number of training samples: {len(train_data)} - test samples: {len(test_data)}')
    print(f'Number of speakers : {len(speaker_counts)} -> {len(speaker_counts_filtered)} (after filtering)')
    print(f"Estimated duration for entire training data: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    print('-'*50)


def get_duration(total_seconds):
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds


def filter_by_speakers(data, min, max = float('inf') ):
    speaker_sample_counts = data["speaker_id"].value_counts()
    filtered_speaker_ids = speaker_sample_counts[(speaker_sample_counts >= min) & (speaker_sample_counts <= max)].index
    filtered_data = data[data["speaker_id"].isin(filtered_speaker_ids)]
    return filtered_data.copy()


def get_speaker_counts(data):
        speaker_counts = defaultdict(int)
        for speaker_id in data["speaker_id"]:
            speaker_counts[speaker_id] += 1
        return speaker_counts



if __name__ == '__main__':

    dataset_path = '/content/Mozilla/'
    train_info = pd.read_csv( dataset_path + '14407271615 14403167110/cv-corpus-13.0-2023-03-09/fa/train.tsv', delimiter='\t')
    test_info = pd.read_csv( dataset_path + '14407271615 14403167110/cv-corpus-13.0-2023-03-09/fa/test.tsv', delimiter='\t')

    # Load train and test data samples
    train_data = load_data(dataset_path, train_info)
    test_data = load_data(dataset_path, test_info)

    test_data = test_data.drop(index=1414)
    #train_data = train_data.iloc[:200]
    test_data = test_data.iloc[:4000]

    # Count the number of speakers in dataset
    speaker_counts = get_speaker_counts(train_data)
    speaker_counts_filtered = get_speaker_counts(train_data)

    # Report dataset statistics
    dataset_report(train_data, test_data, speaker_counts, speaker_counts_filtered)

    # Load previous checkpoint if available
    config = Configs(3)
    path = config.finetune_cp_path
    checkpoint = torch.load(path) if os.path.isfile(path) else None

    # Initialize the trainer
    trainer = Trainer(config, {'train_data':train_data, 'test_data':test_data}, checkpoint)

    # Train and evaluate the model for 50 epochs
    best_model = trainer.train_and_evaluate(50)
