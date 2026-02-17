# utils/preprocessing.py

# utils/preprocessing.py

import pandas as pd
import torch
import re


def load_text_from_csv(path):
    """
    Loads poem.csv and returns tokenized words.
    Assumes a column named 'text'.
    """

    df = pd.read_csv(path)

    # Combine all poems into one big string
    full_text = " ".join(df["text"].astype(str).tolist())

    # Basic cleaning (important for poetry)
    full_text = full_text.lower()
    full_text = re.sub(r"[^a-zA-Z']", " ", full_text)

    words = full_text.split()

    return words


def build_vocab(words):
    vocab = sorted(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return vocab, word_to_idx, idx_to_word


def create_sequences(words, word_to_idx, seq_length):
    sequences = []
    targets = []

    for i in range(len(words) - seq_length):
        seq = words[i:i+seq_length]
        target = words[i+seq_length]

        sequences.append([word_to_idx[w] for w in seq])
        targets.append(word_to_idx[target])

    return torch.tensor(sequences), torch.tensor(targets)
