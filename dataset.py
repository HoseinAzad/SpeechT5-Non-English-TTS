import numpy as np
from tqdm import tqdm
import librosa
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class SpeechDataset(Dataset):

    def __init__(self, data, speaker_model, tokenizer, feature_extractor, tsr = 16e3):
        self.speaker_model = speaker_model
        self.tokenizer = tokenizer
        self.data = data
        self.feature_extractor = feature_extractor
        self.preprocess(tsr)


    def __getitem__(self, item):
        text = self.data.text.values[item]
        tokenized_text = self.data.tokenized_texts.values[item]
        audio_features = self.data.audio_features.values[item]
        s_embedding = self.data.s_embed.values[item]

        return text, tokenized_text, audio_features, s_embedding, item


    def file_to_array(self, path, sampling_rate):
        array, sr = librosa.load(path, sr= sampling_rate)
        return array


    def create_speaker_embedding(self, waveform):
        with torch.no_grad():
            speaker_embeddings = self.speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings


    def __len__(self):
        return len(self.data)


    def preprocess(self, tsr):
        audio_features_list, tokenized_text_list, speaker_embed_list = [], [], []
        for index, row in tqdm(self.data.iterrows(), desc='Preprocessing data') :
            # Convert audio file to waveform arrays and resample it
            wav_array = self.file_to_array(row.path, tsr)
            audio_features_list.append( self.feature_extractor(audio_target= wav_array, sampling_rate= 16e3))

            # Extract speeker embedding vector
            s_embed = self.create_speaker_embedding(wav_array)
            speaker_embed_list.append(s_embed)

            # Normalize text (Optional - converting all homophonic chars to a unique char)
            row.text = self.normalize(row.text)
            tokenized_text_list.append(self.tokenizer(text = ' ' + row.text))

        self.data['tokenized_texts'] = tokenized_text_list
        self.data['audio_features'] = audio_features_list
        self.data['s_embed'] = speaker_embed_list


    def normalize(self, text):
        persian_homophon_chars = {
            'ا':['إ','أ','ئ','ؤ'],
            'ق':['غ'], 'ز':['ظ','ض','ذ'],
            'س':['ص','ث'], 'ت':['ط','ة'],
            'ک':['ك'], '':['ّ', 'ً'], ',':'،'}

        for key in persian_homophon_chars:
            for char in persian_homophon_chars[key]:
                text = text.replace(char, key)
        return text



def get_data_loaders( train_data, test_data, train_bs, tokenizer, feature_extractor, speaker_model, reduction_factor):
    _collate_fn = lambda batch: custom_collate_fn(batch,
                                            feature_extractor = feature_extractor,
                                            tokenizer = tokenizer,
                                            reduction_factor = reduction_factor)

    train_dl = DataLoader(SpeechDataset(train_data, speaker_model, tokenizer, feature_extractor),
                            batch_size=train_bs,
                            collate_fn = _collate_fn,
                            shuffle=True)

    test_dl = DataLoader(SpeechDataset(test_data, speaker_model, tokenizer, feature_extractor),
                            batch_size=8,
                            collate_fn = _collate_fn)

    return train_dl, test_dl


def custom_collate_fn ( batch, tokenizer, feature_extractor, reduction_factor):
    text, tokenized_text, audio_features, speaker_embeddings, items = zip(*batch)

    input_ids = [{"input_ids": feature["input_ids"]} for feature in tokenized_text]
    labels = [{"input_values": feature["input_values"][0]} for feature in audio_features]

    feature_size_hack = feature_extractor.feature_size
    feature_extractor.feature_size = feature_extractor.num_mel_bins
    targets = feature_extractor.pad(labels, return_tensors="pt")
    feature_extractor.feature_size = feature_size_hack
    labels = targets.input_values

    tokenized_text = tokenizer.pad(input_ids, return_tensors="pt")
    input_ids = tokenized_text.input_ids
    attnetion_mask = tokenized_text.attention_mask

    speaker_embeddings = torch.tensor(np.array(speaker_embeddings))
    items = torch.tensor(np.array(items))

    # Replace padding with -100 to ignore loss correctly
    labels = labels.masked_fill( targets.attention_mask.unsqueeze(-1).ne(1), -100)

    # Round down target lengths to multiple of reduction factor
    target_length = labels.shape[1]
    target_length -=  target_length % reduction_factor
    labels = labels[:, :target_length]

    corrected_input_ids = []
    for sequence in input_ids.tolist():
        temp = [id if id!= tokenizer(' ').input_ids[0] else 4 for id in sequence]
        corrected_input_ids.append(temp)
    input_ids = torch.tensor(corrected_input_ids)

    return text, input_ids, labels, speaker_embeddings, attnetion_mask, items