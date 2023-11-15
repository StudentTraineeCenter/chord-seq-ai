import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gain = nn.Linear(d_model, d_model)
        self.bias = nn.Linear(d_model, d_model)

    def forward(self, x, style):
        gain = self.gain(F.relu(style)).unsqueeze(1).repeat(1, x.shape[1], 1)
        bias = self.bias(F.relu(style)).unsqueeze(1).repeat(1, x.shape[1], 1)
        return (1 + gain) * self.norm(x) + bias


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.layer_norm1 = AdaptiveLayerNorm(d_model)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.dropout1 = nn.Dropout(0.1)
        self.layer_norm2 = AdaptiveLayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)
        )
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, style, mask=None):
        # Multi-head attention
        att_input = self.layer_norm1(x, style)
        att_output = self.multi_head_attention(
            att_input, att_input, att_input, attn_mask=mask, need_weights=False
        )[0]
        x = x + self.dropout1(att_output)

        # Feed forward
        ff_input = self.layer_norm2(x, style)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout2(ff_output)

        return x


class ConditionalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, input_len, VOCAB_SIZE, style_dim):
        super().__init__()
        self.style_map = nn.Linear(style_dim, d_model)
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.output = nn.Linear(d_model, VOCAB_SIZE)

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

    def forward(self, x, style):
        x = self.embedding(x)
        style = self.style_map(style)
        x = x + self.pe

        # Generate mask
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Passing through all transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, style, mask)
        x = self.output(x)
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

        self.data = pd.read_csv("../Data/data_tokenized_pitch_class.csv")
        self.all_genres = (
            self.data["genre"].apply(lambda x: x.split("|")).explode().unique()
        )
        self.all_decades = self.data["decade"].unique()
        self.genre_to_idx = {genre: i for i, genre in enumerate(self.all_genres)}
        self.decade_to_idx = {decade: i for i, decade in enumerate(self.all_decades)}

        self.style_dim = len(self.all_genres) + len(self.all_decades)

        ### Get the frequence of each genre and decade
        # Join genre and decade element-wise
        self.joined = self.data["genre"] + "/" + self.data["decade"].astype(str)
        # Get the frequency of each genre/decade
        self.freqs = self.joined.value_counts()
        self.freqs /= self.freqs.sum()

        with open("../Data/token_to_chord.json", "r") as fp:
            token_to_chord = json.load(fp)

        self.VOCAB_SIZE = len(token_to_chord) + 2  # Start and end of sequence tokens

        self.hyperparams = {
            "small": {
                "d_model": 64,
                "n_heads": 8,
                "n_layers": 6,
                "input_len": 256,
                "VOCAB_SIZE": self.VOCAB_SIZE,
                "style_dim": self.style_dim,
            },
            "medium": {
                "d_model": 80,
                "n_heads": 10,
                "n_layers": 12,
                "input_len": 256,
                "VOCAB_SIZE": self.VOCAB_SIZE,
                "style_dim": self.style_dim,
            },
            "large": {
                "d_model": 96,
                "n_heads": 12,
                "n_layers": 24,
                "input_len": 256,
                "VOCAB_SIZE": self.VOCAB_SIZE,
                "style_dim": self.style_dim,
            },
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConditionalTransformer(**self.hyperparams[model_type]).to(
            self.device
        )
        self.model.load_state_dict(
            torch.load(
                f"../Models/ConditionalTransformer{model_type.capitalize()[0]}.pt",
                map_location=self.device,
            )
        )

        self.model.eval()

    def multi_hot(self, genres, decade):
        """Convert a list of genres and a decade into a multi-hot vector."""
        genre_style = torch.zeros((len(self.all_genres)))
        genre_style[[self.genre_to_idx[genre] for genre in genres]] = 1
        genre_style /= genre_style.sum()  # Normalize as there can be multiple genres

        decade_style = torch.zeros((len(self.all_decades)))
        decade_style[self.decade_to_idx[decade]] = 1

        return torch.cat([genre_style, decade_style])

    def _pad(self, chords):
        out = torch.zeros((chords.shape[0], 256), dtype=torch.long)
        out[:, 0] = self.VOCAB_SIZE - 2  # Start of sequence token
        out[:, 1 : chords.shape[-1] + 1] = chords
        return out

    def get_next_probs(self, x, genres, decade):
        """Returns the probability distribution of the next chord given a sequence of chords and the genres and decade.

        Args:
            x (torch.Tensor): A tensor of shape (sequence_length,) containing the sequence of chords. The chords are represented as integers.
            genres (list): A list of genres as strings.
            decade (int): The decade as an integer.

        Returns:
            torch.Tensor: A tensor of shape (VOCAB_SIZE,) containing the probability distribution of the next chord.

        Raises:
            ValueError: If an invalid genre or decade is provided.
        """

        # Ensure that the first chord is not the start of sequence token (it will be added during padding)
        if x[0] == self.VOCAB_SIZE - 2:
            x = x[1:]

        x_len = len(x)
        x = self._pad(x.unsqueeze(0)).to(self.device)

        # Raise an error if the genres or decade are not in the dataset
        invalid_genres = [genre for genre in genres if genre not in self.all_genres]
        if invalid_genres:
            raise ValueError(
                f"Invalid genre(s) {invalid_genres}. Valid genres are {sorted(self.all_genres)}."
            )
        if decade not in self.all_decades:
            raise ValueError(
                f"Invalid decade {decade}. Valid decades are {sorted(self.all_decades)}."
            )

        style = self.multi_hot(genres, decade).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.inference_mode():
            y_pred = self.model(x, style).squeeze()[x_len]

        return F.softmax(y_pred, dim=-1)

    def generate_sequences(self, temperature, max_length, num_sequences=1):
        """Generate a sequence or a batch of sequences of chords. The conditioning is done by sampling from the frequency distribution of genres and decades in the dataset.

        Args:
            temperature (float): The randomness of the generated sequences. A higher temperature means more randomness.
            max_length (int): The maximum length of the generated sequences. It is capped at 254.
            num_sequences (int, optional): The number of sequences to generate. Defaults to 1.

        Returns:
            torch.Tensor: A tensor of shape (num_sequences, max_length + 2) containing the generated sequences.

        Raises:
            ValueError: If an invalid max_length is provided.
        """

        if max_length > 254:
            raise ValueError("The maximum length of the generated sequences is 254.")

        self.model.eval()
        with torch.inference_mode():
            x = torch.zeros(
                (num_sequences, max_length + 2), dtype=torch.long, device=self.device
            )

            styles = self.freqs.sample(num_sequences, replace=True, weights=self.freqs)
            styles = torch.stack(
                [
                    self.multi_hot(
                        styles.index[i].split("/")[0].split("|"),
                        int(styles.index[i].split("/")[1]),
                    )
                    for i in range(num_sequences)
                ]
            ).to(self.device)

            finished_sequences = torch.zeros(
                num_sequences, dtype=torch.bool, device=self.device
            )
            for i in range(max_length):
                input_x = x[:, :i]
                input_x_padded = self._pad(input_x).long()

                y_pred = self.model(input_x_padded.to(self.device), styles).squeeze()[
                    :, i, :
                ]
                # Zero out the probability for the same as the previous chord
                if i > 0:
                    for batch_idx, prev_chord in enumerate(x[:, i - 1]):
                        y_pred[batch_idx, prev_chord] = -torch.inf

                # For the start of sequence token
                y_pred[:, self.VOCAB_SIZE - 2] = -torch.inf

                # Zero out the probability for the end of sequence token
                if i == 0:
                    y_pred[:, self.VOCAB_SIZE - 1] = -torch.inf

                # Sample from the distribution
                y_pred = torch.softmax(y_pred, dim=-1) ** (1 / temperature)
                x[:, i] = y_pred.multinomial(1).squeeze()

                finished_sequences |= x[:, i] == self.VOCAB_SIZE - 1

                if all(finished_sequences):
                    break

            # Ensuring that the last chord in each sequence is VOCAB_SIZE - 1
            x[(x == self.VOCAB_SIZE - 1).sum(dim=1) == 0, -1] = self.VOCAB_SIZE - 1

            # Mask the predictions after the end token
            end_tokens = x.argmax(axis=1).repeat(256, 1).T
            indices = torch.arange(0, 256).repeat(num_sequences, 1).to(self.device)
            x *= end_tokens >= indices

        return x
