import torch
import torch.nn as nn

import json
import numpy as np


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.dropout1 = nn.Dropout(0.1)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)
        )
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Multi-head attention
        att_input = self.layer_norm1(x)
        att_output = self.multi_head_attention(
            att_input, att_input, att_input, attn_mask=mask, need_weights=False
        )[0]
        x = x + self.dropout1(att_output)

        # Feed forward
        ff_input = self.layer_norm2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout2(ff_output)

        return x


class ClassificationTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, input_len, output_dim, VOCAB_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.output = nn.Linear(d_model, output_dim)

        # Positional encoding
        self.register_buffer("pe", torch.zeros(input_len, d_model))
        pos = torch.arange(0, input_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 1::2] = torch.cos(pos * div_term)

    def generate_square_subsequent_mask(self, size):
        """Generate a boolean mask to avoid attending to future tokens."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, x):
        x = self.extract_features(x)

        return self.output(x)

    def extract_features(self, x):
        x = self.embedding(x)
        x = x + self.pe

        # Generate mask
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Passing through all transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        x = x.mean(dim=1)

        return x


class Main:
    VALID_MODEL_TYPES = {"small", "medium", "large"}

    def __init__(self, model_type):
        """
        Initialize the Main class with a specified model type.

        Args:
            model_type (str): The type of the model to initialize. Valid options are 'small', 'medium', 'large'.

        Raises:
            ValueError: If an invalid model_type is provided.
        """

        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Valid options are {self.VALID_MODEL_TYPES}."
            )

        with open("../Data/token_to_chord.json", "r") as fp:
            token_to_chord = json.load(fp)

        self.VOCAB_SIZE = len(
            token_to_chord
        )  # Start and end of sequence tokens are not needed

        self.hyperparams = {
            "small": {
                "d_model": 64,
                "n_heads": 8,
                "n_layers": 6,
                "input_len": 256,
                "output_dim": 28,
                "VOCAB_SIZE": self.VOCAB_SIZE,
            },
            "medium": {
                "d_model": 80,
                "n_heads": 10,
                "n_layers": 6,
                "input_len": 256,
                "output_dim": 28,
                "VOCAB_SIZE": self.VOCAB_SIZE,
            },
            "large": {
                "d_model": 96,
                "n_heads": 12,
                "n_layers": 6,
                "input_len": 256,
                "output_dim": 28,
                "VOCAB_SIZE": self.VOCAB_SIZE,
            },
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClassificationTransformer(**self.hyperparams[model_type]).to(
            self.device
        )
        self.model.load_state_dict(
            torch.load(
                f"../Models/ClassificationTransformer{model_type.capitalize()[0]}.pt",
                map_location=self.device,
            )
        )

        self.model.eval()

    def _pad(self, chords):
        """Pad the input tensor of shape [n] into shape [256] with zeros."""
        out = torch.zeros((256))
        out[: len(chords)] = chords
        return out

    def batch_extract_features(self, chords):
        """Extract features from a batch of padded sequences.

        Args:
            chords (torch.Tensor): A tensor of shape [batch_size, 256] containing the padded sequences.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, d_model] containing the extracted features.
        """
        chords = chords.to(self.device)
        with torch.inference_mode():
            return self.model.extract_features(chords).cpu().numpy()

    def extract_features(self, chords):
        """Extract features from a single input tensor.

        Args:
            chords (torch.Tensor): A tensor of shape [n] containing the sequence of chords.

        Returns:
            torch.Tensor: A tensor of shape [d_model] containing the extracted features.
        """
        chords = self._pad(chords).unsqueeze(0)
        return self.batch_extract_features(chords)[0]
