import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentNet(nn.Module):
    def __init__(self, VOCAB_SIZE):
        super().__init__()
        self.chord_embeddings = nn.Embedding(VOCAB_SIZE, 96)
        self.gru = nn.GRU(96, 96, num_layers=3, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, VOCAB_SIZE),
        )

    def forward(self, x):
        x = self.chord_embeddings(x)
        x, _ = self.gru(x)
        return self.mlp(x)

    def predict_next(self, x):
        x = self.chord_embeddings(x)
        x, _ = self.gru(x)
        return self.mlp(x[:, -1])


class Main:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open("../Data/token_to_chord.json", "r") as fp:
            token_to_chord = json.load(fp)

        self.VOCAB_SIZE = len(token_to_chord) + 2  # Start and end of sequence tokens

        # Load the model
        self.model = RecurrentNet(self.VOCAB_SIZE).to(self.device)
        self.model.load_state_dict(
            torch.load("../Models/RecurrentNet.pt", map_location=self.device)
        )

        self.model.eval()

    def get_next_probs(self, x):
        """Returns the probability distribution of the next chord given a sequence of chords.

        Args:
            x (torch.Tensor): A tensor of shape (sequence_length,) containing the sequence of chords. The chords are represented as integers.

        Returns:
            torch.Tensor: A tensor of shape (VOCAB_SIZE,) containing the probability distribution of the next chord.
        """

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Make sure that the sequence starts with the start of sequence token
        if x[:, 0] != self.VOCAB_SIZE - 2:
            x = torch.cat(
                (torch.tensor([self.VOCAB_SIZE - 2], dtype=torch.long).unsqueeze(0), x),
                dim=-1,
            )

        x = x.to(self.device)

        self.model.eval()
        with torch.inference_mode():
            y_pred = self.model.predict_next(x)

        return F.softmax(y_pred, dim=-1).squeeze()

    def generate_sequences(self, temperature, max_length, num_sequences=1):
        """Generate a sequence or a batch of sequences of chords.

        Args:
            temperature (float): The randomness of the generated sequences. A higher temperature means more randomness.
            max_length (int): The maximum length of the generated sequences. It is capped at 254.
            num_sequences (int, optional): The number of sequences to generate. Defaults to 1.

        Returns:
            torch.Tensor: A tensor of shape (num_sequences, max_length + 2) containing the generated sequences.
        """

        if max_length > 254:
            raise ValueError("The maximum length of the generated sequences is 254.")

        self.model.eval()
        with torch.inference_mode():
            x = torch.zeros(
                (num_sequences, max_length + 2), dtype=torch.long, device=self.device
            )
            # The start of sequence token
            x[:, 0] = self.VOCAB_SIZE - 2

            ended_sequences = torch.zeros(
                num_sequences, dtype=torch.bool, device=self.device
            )
            for i in range(max_length):
                y_pred = self.model.predict_next(x[:, : i + 1])
                # Zero out the probability for the same as the previous chord
                y_pred[:, x[:, i]] = -torch.inf
                # Sample from the distribution
                y_pred = F.softmax(y_pred, dim=-1) ** (1 / temperature)
                x[:, i + 1] = y_pred.multinomial(1).squeeze(1)

                ended_sequences |= x[:, i + 1] == self.VOCAB_SIZE - 1

                if all(ended_sequences):
                    break

            # Ensuring that the last chord in each sequence is VOCAB_SIZE - 1
            x[(x == self.VOCAB_SIZE - 1).sum(dim=1) == 0, -1] = self.VOCAB_SIZE - 1

            # Mask the predictions after the end token
            end_tokens = x.argmax(axis=1).repeat(256, 1).T
            indices = torch.arange(0, 256).repeat(num_sequences, 1).to(self.device)
            x *= end_tokens >= indices

        return x
