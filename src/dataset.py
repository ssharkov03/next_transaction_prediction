from tqdm.auto import trange
from collections import Counter
from typing import List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from einops import rearrange
from torch.utils.data import Dataset, DataLoader

MAX_SEQ_LEN = 512


def tokenize(text):
    """
    Токенизация текста
    """
    x = text.lower()
    x = [y for y in x.split(' ') if y]  # 'if y' removes empty tokens
    x = ['<BOS>'] + x + ['<EOS>']
    return x


def get_feature_preprocess(dataframe, feature_name="mcc", verbose=False, min_freq=3):
    """
    Получение функции для препроцессинга фичи, а также для получения статистики по фиче (например длина словаря).
    """

    print(f"\nCalculating ids for {feature_name}")

    # Посчитаем частоту встречаемости различных токенов (не нужно для bos и eos)
    vocab_freq = Counter()

    # Параллельно заодно посчитаем длины заголовков в токенах (сколько раз встречалась какая длина в токенах)
    sizes = Counter()
    for i in trange(len(dataframe)):
        caption =  dataframe.iloc[i][feature_name]
        tokens = tokenize(caption)
        sizes[len(tokens[1:-1])] += 1
        for token in tokens[1:-1]:
            vocab_freq[token] += 1

    global_max_seq_len = np.max(list(sizes.keys()))

    if verbose:
        show_ = 20
        fig, _ = plt.subplots(3, 1, figsize=(20, 12))
        plt.subplot(312)
        vocab_freq = {k: v for k, v in sorted(vocab_freq.items(), key=lambda item: item[1])}
        plt.title('least popular words')
        plt.bar(list(vocab_freq.keys())[:show_], list(vocab_freq.values())[:show_])

        plt.subplot(311)
        vocab_freq = {k: v for k, v in sorted(vocab_freq.items(), key=lambda item: item[1], reverse=True)}
        plt.title('most popular words')
        plt.bar(list(vocab_freq.keys())[:show_], list(vocab_freq.values())[:show_])

        plt.subplot(313)
        plt.title('sequence sizes')
        plt.bar(list(sizes.keys()), list(sizes.values()))

        fig.tight_layout()
        plt.show()

    MIN_FREQ = min_freq  # токены с частотой ниже этой константы заменяются на <UNK>

    # Так же добавляем <PAD> токен для паддингов
    tok_to_ind = {
        '<UNK>': 0,
        '<BOS>': 1,
        '<EOS>': 2,
        '<PAD>': 3,
    }

    ind_to_tok = {
        0: '<UNK>',
        1: '<BOS>',
        2: '<EOS>',
        3: '<PAD>',
    }

    # Заполнить оставшееся
    for token, freq in vocab_freq.items():
        if freq >= MIN_FREQ:
            token_ind = len(tok_to_ind)
            tok_to_ind[token] = token_ind
            ind_to_tok[token_ind] = token


    assert len(tok_to_ind) == len(ind_to_tok)  
    vocab_size = len(tok_to_ind)
    print(f"Resulting vocab size: {vocab_size} (out of {len(vocab_freq) + 4} tokens overall, due to MIN_FREQ={MIN_FREQ})")

    # Функция возвращает по тексту индексы токенов в тексте
    def to_ids(text):
        tokens = tokenize(text)
        ids = [tok_to_ind.get(x, tok_to_ind['<UNK>']) for x in tokens]
        return ids

    return to_ids, vocab_size, tok_to_ind, global_max_seq_len


def get_mappings(train_data, features_list):
    """
    Возвращает 2 словаря:
    1) feature2preprocess - по названию фичи сможем получать препроцессинг функцию (для перевода текста в индексы токенов)
    2) feature2vocab_size - по названию фичи сможем получать длину словаря для фичи
    """
    if isinstance(train_data, str):
        train_data = preprocess_df(train_data)

    feature2preprocess = dict()
    feature2vocab_size = dict()
    for feature_name in features_list:
        to_ids, vocab_size, tok_to_ind, global_max_seq_len = get_feature_preprocess(train_data, feature_name=feature_name)
        feature2preprocess[feature_name] = to_ids  
        feature2vocab_size[feature_name] = vocab_size 
    return feature2preprocess, feature2vocab_size


def preprocess_df(data_path):
    """
    Базовый препроцессинг датасета
    """
    data = pd.read_parquet(data_path)
    data = data[["app_id", "mcc", "amnt", "day_of_week", "payment_system"]]
    
    # Группировка по столбцу "app_id" и объединение значений в остальных столбцах
    data = data.groupby('app_id').agg({
        'mcc': lambda x: " ".join([str(y) for y in list(x)]),
        'amnt': lambda x: " ".join([str(y) for y in list(x)]),
        'day_of_week': lambda x: " ".join([str(y) for y in list(x)]),
        'payment_system': lambda x: " ".join([str(y) for y in list(x)])
    }).reset_index()
    return data


class SequenceDataset(Dataset):
    def __init__(self, data_path, feature2preprocess, train=False):
        super(SequenceDataset, self).__init__()

        self.df = preprocess_df(data_path)
        self.feature2preprocess = feature2preprocess
        self.features_list = feature2preprocess.keys() 
        self.train = train

    def __getitem__(self, index):
        features_inds = []
        for feature_name in self.features_list:
            feature_values = self.df.iloc[index][feature_name]
            features_inds.append(self.feature2preprocess[feature_name](feature_values))
        return features_inds
    
    def __len__(self):
        return len(self.df)
    

def pad_sequence(
    sequence_inds: List[int],
    max_seq_len: int,
    value: int,
):
    """
    Функция для паддинга последовательности.
    """
    padded_caption = torch.full((max_seq_len,), value)
    padded_caption[:len(sequence_inds)] = torch.LongTensor(sequence_inds)
    return padded_caption


def collate_fn(batch):
    sequences_batch_list = []
    local_max_seq_len = np.max([len(feature_ids) for features_ids in batch for feature_ids in features_ids])
    
    for features_sequences in batch:
        padded_sequences_list = [pad_sequence(
            feature_sequence, 
            local_max_seq_len, 
            3, # const ind of <PAD> token
        ).unsqueeze(0) for feature_sequence in features_sequences]
        padded_sequences = torch.cat(padded_sequences_list, dim=0)
        sequences_batch_list.append(padded_sequences.unsqueeze(0))

    sequences_batch = torch.cat(sequences_batch_list, dim=0)
    sequences_batch = rearrange(sequences_batch, "bs feats seq -> bs seq feats") 
    sequences_batch = sequences_batch[:, -MAX_SEQ_LEN:, ...]
    return sequences_batch

def create_dataloaders(features_list, train_data_path, val_data_path):
    """
    Создание даталодеров и словаря с длинами словарей по фичам.
    """
    train_df = preprocess_df(train_data_path)
    feature2preprocess, feature2vocab_size = get_mappings(train_df, features_list)
    
    train_ds = SequenceDataset(data_path=train_data_path, feature2preprocess=feature2preprocess, train=True)
    val_ds = SequenceDataset(data_path=val_data_path, feature2preprocess=feature2preprocess)

    batch_size = 16
    num_workers = 0

    dataloader_train = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_val = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return dataloader_train, dataloader_val, feature2vocab_size