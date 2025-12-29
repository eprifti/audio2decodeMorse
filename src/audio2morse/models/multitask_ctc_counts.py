import torch
import torch.nn as nn
import torch.nn.functional as F

from audio2morse.models.multitask_ctc import ConvBlock


class MultiTaskCTCCountsModel(nn.Module):
    """
    Multi-task model:
      - CTC text head (as usual).
      - Bit/gap heads (dot/dash/blank, gap classes) inherited from the multitask variant.
      - Character count head (regression to total characters incl. spaces).
      - Character histogram head (bag-of-characters multi-label logits).
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
        use_mask_head: bool = False,
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
        self.bit_head = nn.Linear(rnn_out_dim, 3)
        self.gap_head = nn.Linear(rnn_out_dim, 3)
        self.count_head = nn.Linear(rnn_out_dim, 1)  # predict total characters
        self.char_hist_head = nn.Linear(rnn_out_dim, vocab_size - 1)  # exclude blank
        self.use_mask_head = use_mask_head
        if self.use_mask_head:
            self.mask_head = nn.Linear(rnn_out_dim, 1)  # framewise on/off logits

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        x = features.unsqueeze(1)  # (B,1,T,F)
        x = self.convs(x)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.proj(x)

        out_lengths = torch.div(lengths, self.downsample, rounding_mode="floor")
        packed = nn.utils.rnn.pack_padded_sequence(
            x, out_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        text_log_probs = F.log_softmax(self.text_head(enc), dim=-1)  # (B,T',V)
        bit_logits = self.bit_head(enc)
        gap_logits = self.gap_head(enc)

        # For count/hist we use a mean-pooled summary over time
        summary = enc.mean(dim=1)  # (B, rnn_out_dim)
        count_pred = self.count_head(summary).squeeze(-1)  # (B,)
        hist_logits = self.char_hist_head(summary)  # (B, V-1)

        out = {
            "text_log_probs": text_log_probs,
            "bit_logits": bit_logits,
            "gap_logits": gap_logits,
            "count_pred": count_pred,
            "hist_logits": hist_logits,
            "out_lengths": out_lengths,
        }
        if self.use_mask_head:
            out["mask_logits"] = self.mask_head(enc).squeeze(-1)
        return out
