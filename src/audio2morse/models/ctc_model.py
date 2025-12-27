from typing import List, Sequence

import torch
import torch.nn as nn


class CTCMorseModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        cnn_channels: List[int],
        pool_kernel: Sequence[Sequence[int]] = None,
        rnn_hidden_size: int,
        rnn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        convs = []
        in_ch = 1
        if pool_kernel is None:
            pool_kernel = [[2, 2] for _ in cnn_channels]
        time_factor = 1
        freq_factor = 1
        for idx, ch in enumerate(cnn_channels):
            convs.append(nn.Conv2d(in_ch, ch, kernel_size=(3, 3), padding=1))
            convs.append(nn.BatchNorm2d(ch))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout2d(p=dropout))
            k_t, k_f = pool_kernel[idx] if idx < len(pool_kernel) else (2, 2)
            convs.append(nn.MaxPool2d(kernel_size=(k_t, k_f)))
            time_factor *= k_t
            freq_factor *= k_f
            in_ch = ch
        self.cnn = nn.Sequential(*convs)
        self.time_pool_factor = time_factor
        self.freq_pool_factor = freq_factor

        projected_dim = (input_dim // freq_factor) * cnn_channels[-1]
        self.proj = nn.Sequential(
            nn.Linear(projected_dim, rnn_hidden_size),
            nn.LayerNorm(rnn_hidden_size),
        )
        self.rnn = nn.LSTM(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, rnn_hidden_size),
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
