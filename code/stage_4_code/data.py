# src/data.py

import os
import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import string
from src.config import CLASS_DIR, CLS_MAX_LEN


class TextClassificationDataset(Dataset):
    def __init__(self, split: str, min_freq: int = 2):
        """
        Improved dataset with better preprocessing
        """
        # 1) Read and preprocess text
        texts, labels = [], []

        for label in ("neg", "pos"):
            folder = os.path.join(CLASS_DIR, split, label)
            if not os.path.exists(folder):
                raise ValueError(f"Folder {folder} not found!")

            files = [f for f in os.listdir(folder) if f.endswith('.txt')]
            print(f"Loading {len(files)} {label} files from {split} set...")

            for fn in files:
                file_path = os.path.join(folder, fn)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()

                    # Preprocess text
                    tokens = self.preprocess_text(text)

                    if len(tokens) > 0:  # Only keep non-empty reviews
                        texts.append(tokens)
                        labels.append(0 if label == "neg" else 1)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

        print(f"Loaded {len(texts)} valid samples")

        # 2) Build vocabulary
        self.build_vocab(texts, min_freq)

        # 3) Convert to examples
        self.examples = []
        for tokens, lbl in zip(texts, labels):
            # Convert tokens to indices
            ids = [self.token_to_idx.get(tok, self.token_to_idx['<unk>'])
                   for tok in tokens[:CLS_MAX_LEN]]

            if len(ids) > 0:  # Safety check
                length = len(ids)
                self.examples.append((
                    torch.tensor(ids, dtype=torch.long),
                    length,
                    torch.tensor(lbl, dtype=torch.long)
                ))

        print(f"Created {len(self.examples)} examples")

    def preprocess_text(self, text):
        """
        Improved text preprocessing
        """
        # Convert to lowercase
        text = text.lower()

        # Replace HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Handle contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        # Remove special characters but keep some punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)

        # Tokenize by splitting on whitespace
        tokens = text.split()

        # Remove very short tokens (likely noise)
        tokens = [tok for tok in tokens if len(tok) > 1 or tok in ['i', 'a']]

        return tokens

    def build_vocab(self, texts, min_freq):
        """Build vocabulary from texts"""
        counter = Counter()
        for tokens in texts:
            counter.update(tokens)

        # Special tokens
        self.token_to_idx = {
            '<pad>': 0,
            '<unk>': 1,
            '<start>': 2,
            '<end>': 3
        }

        # Add frequent tokens
        for token, count in counter.most_common():
            if count >= min_freq:
                self.token_to_idx[token] = len(self.token_to_idx)

        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        print(f"Vocabulary size: {len(self.token_to_idx)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def cls_collate_fn(batch):
    """
    Improved collate function with dynamic padding
    """
    ids_list, lengths, labels = zip(*batch)

    # Convert lengths to tensor
    lengths = torch.tensor(lengths, dtype=torch.long)

    # Sort by length (descending) for efficient packing
    sorted_indices = lengths.argsort(descending=True)

    # Reorder batch
    ids_list = [ids_list[i] for i in sorted_indices]
    lengths = lengths[sorted_indices]
    labels = torch.stack([labels[i] for i in sorted_indices])

    # Pad sequences
    padded = pad_sequence(ids_list, batch_first=True, padding_value=0)

    return padded, lengths, labels


# Additional preprocessing utilities
def clean_text_for_generation(text):
    """Clean text for generation tasks"""
    # Convert to lowercase
    text = text.lower()

    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Add spaces around punctuation for better tokenization
    for punct in '.,!?;:':
        text = text.replace(punct, f' {punct} ')

    # Remove multiple spaces
    text = ' '.join(text.split())

    return text


def augment_text(text, aug_prob=0.1):
    """
    Simple text augmentation for training

    Args:
        text: list of tokens
        aug_prob: probability of augmenting each token

    Returns:
        augmented text
    """
    import random

    augmented = []
    for token in text:
        if random.random() < aug_prob:
            # Random augmentation strategies
            choice = random.choice(['delete', 'duplicate', 'shuffle'])

            if choice == 'delete':
                continue  # Skip this token
            elif choice == 'duplicate':
                augmented.extend([token, token])
            else:  # shuffle - just add normally
                augmented.append(token)
        else:
            augmented.append(token)

    return augmented