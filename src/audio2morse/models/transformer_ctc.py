import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, d_model)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ConvBlock(nn.Module):
    """Conv + BN + ReLU + MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerCTCMorseModel(nn.Module):
    """
    CNN front-end + Transformer encoder for CTC decoding.
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        cnn_channels: list[int],
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        convs = []
        in_ch = 1
        for ch in cnn_channels:
            convs.append(ConvBlock(in_ch, ch, pool=(2, 2)))
            in_ch = ch
        self.convs = nn.Sequential(*convs)

        self.downsample = 2 ** len(cnn_channels)
        proj_dim = (input_dim // self.downsample) * cnn_channels[-1]
        self.input_proj = nn.Linear(proj_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, F)
            lengths:  (B,)
        Returns:
            Log-probs (T', B, V) for CTC
        """
        x = features.unsqueeze(1)  # (B,1,T,F)
        x = self.convs(x)  # (B,C,T',F')
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)  # (B,T',C*F')
        x = self.input_proj(x)  # (B,T',d_model)

        # Lengths after downsample
        out_lengths = torch.div(lengths, self.downsample, rounding_mode="floor")

        # Transformer expects (T,B,E)
        x = x.transpose(0, 1)  # (T',B,d_model)
        x = self.pos_enc(x)

        # Padding mask: True for pads
        max_t = x.size(0)
        pad_mask = torch.arange(max_t, device=out_lengths.device).expand(len(out_lengths), max_t) >= out_lengths.unsqueeze(1)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        logits = self.classifier(x)  # (T',B,V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
