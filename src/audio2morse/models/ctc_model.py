from typing import List

import torch
import torch.nn as nn


class CTCMorseModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        cnn_channels: List[int],
        rnn_hidden_size: int,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        convs = []
        in_ch = 1
        for ch in cnn_channels:
            convs.append(nn.Conv2d(in_ch, ch, kernel_size=(3, 3), padding=1))
            convs.append(nn.BatchNorm2d(ch))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout2d(p=dropout))
            convs.append(nn.MaxPool2d(kernel_size=(2, 2)))
            in_ch = ch
        self.cnn = nn.Sequential(*convs)

        projected_dim = (input_dim // (2 ** len(cnn_channels))) * cnn_channels[-1]
        self.proj = nn.Sequential(
            nn.Linear(projected_dim, rnn_hidden_size),
            nn.LayerNorm(rnn_hidden_size),
        )
        self.rnn = nn.LSTM(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        rnn_out_dim = rnn_hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, rnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_size, vocab_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, time, mel)
        Returns:
            log_probs: (time, batch, vocab)
        """
        x = features.unsqueeze(1)  # (B,1,T,F)
        x = self.cnn(x)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.proj(x)
        x, _ = self.rnn(x)
        logits = self.classifier(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)  # CTC expects (T, B, V)
