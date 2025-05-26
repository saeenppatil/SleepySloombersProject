# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ImprovedRNNClassifier(nn.Module):
    """Enhanced RNN classifier with multiple improvements for better accuracy"""

    def __init__(
            self,
            vocab_size: int,
            emb_dim: int,
            hidden_dim: int,
            pad_idx: int,
            num_layers: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = True,
            num_classes: int = 2,
            rnn_type: str = "LSTM"  # Can be "RNN", "LSTM", or "GRU"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)

        # Choose RNN type
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:  # vanilla RNN
            self.rnn = nn.RNN(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )

        # Attention mechanism
        self.use_attention = True
        if self.use_attention:
            self.attention = nn.Linear(
                hidden_dim * (2 if bidirectional else 1),
                1
            )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        factor = 2 if bidirectional else 1

        # Add a hidden layer before final classification
        self.fc1 = nn.Linear(hidden_dim * factor, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * factor)

    def attention_weights(self, rnn_output, lengths):
        """Compute attention weights for each sequence"""
        # rnn_output: [B, L, H*2]
        attention_scores = self.attention(rnn_output)  # [B, L, 1]
        attention_scores = attention_scores.squeeze(-1)  # [B, L]

        # Create mask for padding
        batch_size, max_len = attention_scores.size()
        mask = torch.arange(max_len).expand(batch_size, max_len).to(lengths.device)
        mask = mask < lengths.unsqueeze(1)

        # Apply mask
        attention_scores = attention_scores.masked_fill(~mask, -float('inf'))
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, L]

        return attention_weights

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)

        # 1) Embed and dropout
        emb = self.embedding(input_ids)  # [B, L, E]
        emb = self.emb_dropout(emb)

        # 2) Pack sequences
        packed = pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # 3) Run through RNN
        if self.rnn_type == "LSTM":
            packed_out, (h_n, c_n) = self.rnn(packed)
        else:
            packed_out, h_n = self.rnn(packed)

        # 4) Unpack to get all hidden states
        rnn_output, _ = pad_packed_sequence(packed_out, batch_first=True)
        # rnn_output: [B, L, H*2] if bidirectional

        # 5) Apply attention or use last hidden state
        if self.use_attention and self.training:
            # Attention mechanism
            attention_weights = self.attention_weights(rnn_output, lengths)  # [B, L]
            attended = torch.bmm(
                attention_weights.unsqueeze(1),
                rnn_output
            ).squeeze(1)  # [B, H*2]
            h = attended
        else:
            # Use last hidden state
            if self.bidirectional:
                h_fwd = h_n[-2]  # [B, H]
                h_bwd = h_n[-1]  # [B, H]
                h = torch.cat([h_fwd, h_bwd], dim=1)  # [B, H*2]
            else:
                h = h_n[-1]  # [B, H]

        # 6) Layer norm
        h = self.layer_norm(h)

        # 7) Dropout and classification
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        logits = self.fc2(h)

        return logits  # [B, num_classes]


class RNNGenerator(nn.Module):
    """RNN model for text generation"""

    def __init__(
            self,
            vocab_size: int,
            emb_dim: int,
            hidden_dim: int,
            num_layers: int = 2,
            dropout: float = 0.3,
            tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # RNN layer (GRU typically works well for generation)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Optionally tie weights
        if tie_weights and emb_dim == hidden_dim:
            self.fc.weight = self.embedding.weight

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights with Xavier uniform"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x, hidden=None):
        """
        x: [B, L] token indices
        hidden: previous hidden state or None
        """
        # Embed and dropout
        emb = self.embedding(x)  # [B, L, E]
        emb = self.emb_dropout(emb)

        # Run through RNN
        output, hidden = self.rnn(emb, hidden)  # output: [B, L, H]

        # Apply dropout and projection
        output = self.dropout(output)
        logits = self.fc(output)  # [B, L, V]

        return logits, hidden

    def generate(self, start_tokens, max_length=100, temperature=1.0, device='cpu'):
        """
        Generate text given starting tokens

        Args:
            start_tokens: list of token indices
            max_length: maximum generation length
            temperature: sampling temperature (higher = more random)
            device: torch device

        Returns:
            generated: list of token indices
        """
        self.eval()
        generated = start_tokens.copy()
        hidden = None

        with torch.no_grad():
            for _ in range(max_length - len(start_tokens)):
                # Get last token
                x = torch.tensor([generated[-1]], device=device).unsqueeze(0)  # [1, 1]

                # Forward pass
                logits, hidden = self.forward(x, hidden)  # logits: [1, 1, V]
                logits = logits[0, -1, :] / temperature  # [V]

                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                generated.append(next_token)

                # Stop if we hit end token (if you have one)
                if next_token == 0:  # Assuming 0 is padding/end
                    break

        return generated

    def beam_search(self, start_tokens, beam_width=5, max_length=100, device='cpu'):
        """
        Generate text using beam search for better quality
        """
        self.eval()

        # Initialize beams
        beams = [(start_tokens.copy(), 0.0, None)]  # (sequence, score, hidden)

        with torch.no_grad():
            for _ in range(max_length - len(start_tokens)):
                new_beams = []

                for seq, score, hidden in beams:
                    x = torch.tensor([seq[-1]], device=device).unsqueeze(0)
                    logits, new_hidden = self.forward(x, hidden)
                    logits = logits[0, -1, :]  # [V]

                    # Get top-k tokens
                    log_probs = F.log_softmax(logits, dim=-1)
                    topk_log_probs, topk_indices = log_probs.topk(beam_width)

                    for log_prob, idx in zip(topk_log_probs, topk_indices):
                        new_seq = seq + [idx.item()]
                        new_score = score + log_prob.item()
                        new_beams.append((new_seq, new_score, new_hidden))

                # Keep top beam_width beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]

                # Check if all beams have ended
                if all(b[0][-1] == 0 for b in beams):
                    break

        # Return best sequence
        return beams[0][0]


# Keep the original simple versions for backward compatibility
class RNNClassifier(nn.Module):
    """Original RNN classifier (kept for compatibility)"""

    def __init__(
            self,
            vocab_size: int,
            emb_dim: int,
            hidden_dim: int,
            pad_idx: int,
            num_layers: int = 1,
            dropout: float = 0.1,
            bidirectional: bool = True,
            num_classes: int = 2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * factor, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        packed = pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, h_n = self.rnn(packed)

        if self.rnn.bidirectional:
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            h = torch.cat([h_fwd, h_bwd], dim=1)
        else:
            h = h_n[-1]

        out = self.dropout(h)
        return self.fc(out)