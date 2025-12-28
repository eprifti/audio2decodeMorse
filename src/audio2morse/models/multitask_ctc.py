import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Small conv block with BN/ReLU/MaxPool for feature downsampling."""

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


class MultiTaskCTCMorseModel(nn.Module):
    """
    Prototype multi-task model:
      - Shared CNN encoder over log-mel features.
      - Shared RNN encoder (LSTM/biLSTM).
      - Text head for CTC logits (blank-inclusive).
      - Bit head for framewise dot/dash/blank classification.
      - Gap head for framewise gap type (none / char-gap / word-gap).
    The auxiliary heads are meant to help the model learn timing/segmentation cues.
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        cnn_channels: list[int] = [32, 64],
        rnn_hidden_size: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
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
        self.proj = nn.Linear(proj_dim, rnn_hidden_size)

        self.rnn = nn.LSTM(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        rnn_out_dim = rnn_hidden_size * (2 if bidirectional else 1)

        # Heads
        self.text_head = nn.Linear(rnn_out_dim, vocab_size)
        self.bit_head = nn.Linear(rnn_out_dim, 3)  # dot / dash / blank
        self.gap_head = nn.Linear(rnn_out_dim, 3)  # none / char-gap / word-gap

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, F) log-mel input
            lengths:  (B,) original frame lengths before conv downsampling
        Returns:
            dict with text_logits, bit_logits, gap_logits, out_lengths
        """
        x = features.unsqueeze(1)  # (B,1,T,F)
        x = self.convs(x)  # (B,C,T',F')
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)  # (B,T',C*F')
        x = self.proj(x)

        # Adjust lengths for downsampling
        out_lengths = torch.div(lengths, self.downsample, rounding_mode="floor")

        packed = nn.utils.rnn.pack_padded_sequence(
            x, out_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        text_log_probs = F.log_softmax(self.text_head(enc), dim=-1)  # (B, T', vocab)
        bit_logits = self.bit_head(enc)    # (B, T', 3)
        gap_logits = self.gap_head(enc)    # (B, T', 3)

        return {
            "text_log_probs": text_log_probs,
            "bit_logits": bit_logits,
            "gap_logits": gap_logits,
            "out_lengths": out_lengths,
        }
