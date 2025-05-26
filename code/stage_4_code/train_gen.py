# src/train_gen.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import json
import csv
from collections import Counter

from src.config import (
    GEN_FILE, GEN_MAX_LEN,
    GEN_EMB_DIM, GEN_HIDDEN_DIM, GEN_NUM_LAYERS, GEN_DROPOUT,
    GEN_BATCH_SIZE, GEN_LR, GEN_EPOCHS
)
from src.models import RNNGenerator

# Import GloVe loader from train_class
try:
    from src.train_class import load_glove_embeddings
except ImportError:
    print("Warning: Could not import load_glove_embeddings")
    load_glove_embeddings = None


class JokesDataset(Dataset):
    """Dataset for joke generation from CSV file"""

    def __init__(self, file_path, max_len=100, min_freq=2):
        self.max_len = max_len

        # Read jokes from CSV
        print(f"Loading jokes from {file_path}...")
        self.jokes = []

        # Handle both .csv and plain text files
        if file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    joke = row.get('Joke', '').strip().lower()
                    if joke:
                        self.jokes.append(joke)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.jokes = [line.strip().lower() for line in f if line.strip()]

        print(f"Loaded {len(self.jokes)} jokes")

        # Build vocabulary
        self.build_vocab(min_freq)

        # Tokenize and filter
        self.tokenized_jokes = []
        for joke in self.jokes:
            tokens = self.tokenize(joke)
            if len(tokens) > 3:  # Only keep jokes with more than 3 words
                self.tokenized_jokes.append(tokens)

        print(f"Kept {len(self.tokenized_jokes)} jokes after filtering")

        # Create training examples
        self.examples = []
        for tokens in self.tokenized_jokes:
            # Add start and end tokens
            tokens = ['<start>'] + tokens + ['<end>']

            # Create multiple training examples from each joke
            for i in range(1, min(len(tokens) - 1, max_len)):
                input_seq = tokens[:i]
                target_seq = tokens[1:i + 1]
                self.examples.append((input_seq, target_seq))

        print(f"Created {len(self.examples)} training examples")

    def build_vocab(self, min_freq):
        """Build vocabulary from jokes"""
        counter = Counter()
        for joke in self.jokes:
            tokens = self.tokenize(joke)
            counter.update(tokens)

        # Special tokens
        self.token_to_idx = {
            '<pad>': 0,
            '<unk>': 1,
            '<start>': 2,
            '<end>': 3
        }

        # Add frequent tokens
        for token, count in counter.items():
            if count >= min_freq:
                self.token_to_idx[token] = len(self.token_to_idx)

        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)
        print(f"Vocabulary size: {self.vocab_size}")

    def tokenize(self, text):
        """Simple word tokenization"""
        import re
        # Better tokenization for jokes - preserve punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens

    def encode(self, tokens):
        """Convert tokens to indices"""
        return [self.token_to_idx.get(t, 1) for t in tokens]  # 1 is <unk>

    def decode(self, indices):
        """Convert indices back to tokens"""
        return [self.idx_to_token.get(idx, '<unk>') for idx in indices]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_tokens, target_tokens = self.examples[idx]
        input_ids = self.encode(input_tokens)
        target_ids = self.encode(target_tokens)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )


def collate_fn(batch):
    """Pad and batch sequences"""
    inputs, targets = zip(*batch)
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)

        # Reshape for loss calculation
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(logits, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def generate_text(model, dataset, start_text, max_length=50,
                  temperature=0.8, device='cpu'):
    """Generate text from starting words with improved sampling"""
    model.eval()

    # Tokenize starting text
    tokens = dataset.tokenize(start_text.lower())
    if len(tokens) == 0:
        tokens = ['<start>']

    # Encode tokens
    token_ids = dataset.encode(tokens)
    generated_ids = token_ids.copy()

    hidden = None

    with torch.no_grad():
        for _ in range(max_length):
            # Get last token
            x = torch.tensor([generated_ids[-1]], device=device).unsqueeze(0)

            # Forward pass
            logits, hidden = model.forward(x, hidden)
            logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            top_k = 50
            top_k_logits, top_k_indices = torch.topk(logits, top_k)

            # Sample from top-k
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1).item()
            next_token = top_k_indices[next_token_idx].item()

            # Stop if we hit end token
            if next_token == dataset.token_to_idx.get('<end>', 0):
                break

            generated_ids.append(next_token)

    # Decode
    generated_tokens = dataset.decode(generated_ids)

    # Clean up and join tokens
    result = []
    for i, token in enumerate(generated_tokens):
        if token in ['<start>', '<end>', '<pad>']:
            continue
        if i == 0 or token in '.,!?;:':
            result.append(token)
        else:
            result.append(' ' + token)

    return ''.join(result)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check if file exists and has correct extension
    gen_file_path = GEN_FILE
    if not gen_file_path.endswith('.csv') and os.path.exists(GEN_FILE + '.csv'):
        gen_file_path = GEN_FILE + '.csv'

    if not os.path.exists(gen_file_path):
        print(f"Error: File {gen_file_path} not found!")
        return

    # Load dataset
    dataset = JokesDataset(gen_file_path, max_len=GEN_MAX_LEN, min_freq=2)

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=GEN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for compatibility
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=GEN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Create model
    model = RNNGenerator(
        vocab_size=dataset.vocab_size,
        emb_dim=GEN_EMB_DIM,
        hidden_dim=GEN_HIDDEN_DIM,
        num_layers=GEN_NUM_LAYERS,
        dropout=GEN_DROPOUT
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load GloVe embeddings if available
    glove_path = os.path.join("data", "glove", "glove.6B.100d.txt")
    if os.path.exists(glove_path) and GEN_EMB_DIM == 100 and load_glove_embeddings:
        print("Loading GloVe embeddings...")
        emb_weights = load_glove_embeddings(glove_path, dataset.token_to_idx, GEN_EMB_DIM)
        model.embedding.weight.data.copy_(emb_weights)
        model.embedding.weight.requires_grad = False  # Freeze initially
        print("✓ GloVe embeddings loaded and frozen")
    else:
        print(f"Note: GloVe embeddings not loaded (check dimensions match: {GEN_EMB_DIM} vs 100)")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.AdamW(model.parameters(), lr=GEN_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(1, GEN_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{GEN_EPOCHS}")
        print("-" * 40)

        # Unfreeze embeddings after 5 epochs
        if epoch == 6 and hasattr(model, 'embedding'):
            model.embedding.weight.requires_grad = True
            print("✓ Unfreezing embeddings for fine-tuning")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_generator.pth')
            print("✓ Saved best model")

        # Generate samples every 5 epochs
        if epoch % 5 == 0:
            print("\n--- Sample Generations ---")
            test_starts = [
                "what did the",
                "why did the",
                "how do you",
                "a man walks",
                "knock knock who"
            ]

            for start in test_starts:
                generated = generate_text(
                    model, dataset, start,
                    max_length=30, temperature=0.8, device=device
                )
                print(f"'{start}' → {generated}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    model.load_state_dict(torch.load('best_generator.pth'))
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Generate examples from dataset
    print("\n--- Generations from Dataset Starts ---")
    for i in range(10):
        joke = random.choice(dataset.jokes)
        tokens = dataset.tokenize(joke)
        if len(tokens) >= 3:
            start_tokens = ' '.join(tokens[:3])
            generated = generate_text(
                model, dataset, start_tokens,
                max_length=50, temperature=0.7, device=device
            )
            print(f"\nOriginal: {joke}")
            print(f"Generated: {generated}")

    # Generate from custom starts
    print("\n--- Custom Generations ---")
    custom_starts = [
        "why is the",
        "what happens when",
        "did you hear",
        "a doctor and",
        "my wife told"
    ]

    for start in custom_starts:
        generated = generate_text(
            model, dataset, start,
            max_length=40, temperature=0.8, device=device
        )
        print(f"\nStart: '{start}'")
        print(f"Generated: {generated}")

    # Save results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'vocab_size': dataset.vocab_size,
        'best_val_loss': best_val_loss
    }

    with open('generation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot loss curves
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('generation_loss_curves.png')
        print("\n✓ Saved loss curves to generation_loss_curves.png")
    except:
        print("\nCould not plot loss curves (matplotlib not available)")

    print("\n✓ Training complete!")


# Add the missing import
import torch.nn.functional as F

if __name__ == "__main__":
    main()