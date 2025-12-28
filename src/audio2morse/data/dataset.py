import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

@dataclass
class Sample:
    path: Path
    text: str
    freq_hz: Optional[float] = None
    wpm: Optional[float] = None
    amplitude: Optional[float] = None


def _load_manifest(manifest_path: Path) -> List[Sample]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    samples: List[Sample] = []
    with manifest_path.open("r") as fp:
        for line in fp:
            if not line.strip():
                continue
            entry = json.loads(line)
            samples.append(
                Sample(
                    path=Path(entry["audio_filepath"]),
                    text=entry["text"],
                    freq_hz=entry.get("freq_hz"),
                    wpm=entry.get("wpm"),
                    amplitude=entry.get("amplitude"),
                )
            )
    return samples


class MorseAudioDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: int,
        n_mels: int,
        frame_length_ms: int,
        frame_step_ms: int,
        label_map: Dict[str, int],
        max_duration_s: float = 12.0,
        augment: Optional[Dict] = None,
    ) -> None:
        self.samples = _load_manifest(Path(manifest_path))
        self.sample_rate = sample_rate
        self.max_duration_s = max_duration_s
        self.label_map = label_map
        self.blank_idx = max(label_map.values())
        aug_cfg = augment or {}
        spec_cfg = aug_cfg.get("specaugment", {})
        self.use_specaugment = spec_cfg.get("enabled", False)
        wav_cfg = aug_cfg.get("waveform", {})
        self.use_waveform_aug = wav_cfg.get("enabled", False)
        self.noise_std = float(wav_cfg.get("noise_std", 0.0))
        self.gain_min = float(wav_cfg.get("gain_min", 1.0))
        self.gain_max = float(wav_cfg.get("gain_max", 1.0))

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(sample_rate * frame_length_ms / 1000),
            hop_length=int(sample_rate * frame_step_ms / 1000),
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        if self.use_specaugment:
            freq_param = int(spec_cfg.get("freq_mask_param", 8))
            time_param = int(spec_cfg.get("time_mask_param", 30))
            num_masks = int(spec_cfg.get("num_masks", 2))
            self.freq_masks = [
                torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_param) for _ in range(num_masks)
            ]
            self.time_masks = [
                torchaudio.transforms.TimeMasking(time_mask_param=time_param) for _ in range(num_masks)
            ]
        else:
            self.freq_masks = []
            self.time_masks = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        audio, sr = sf.read(sample.path, always_2d=True)
        waveform = torch.from_numpy(audio.transpose()).float()  # (channels, samples)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        max_samples = int(self.max_duration_s * self.sample_rate)
        waveform = waveform[:, :max_samples]

        # Lightweight waveform augmentation: random gain and additive noise.
        if self.use_waveform_aug:
            if self.gain_min != 1.0 or self.gain_max != 1.0:
                gain = torch.empty(1).uniform_(self.gain_min, self.gain_max).item()
                waveform = waveform * gain
            if self.noise_std > 0:
                noise = torch.randn_like(waveform) * self.noise_std
                waveform = waveform + noise

        with torch.no_grad():
            spec = self.to_db(self.mel(waveform)).squeeze(0)  # (mel, time)
            if self.use_specaugment:
                for fm in self.freq_masks:
                    spec = fm(spec)
                for tm in self.time_masks:
                    spec = tm(spec)
            mel = spec.transpose(0, 1)  # (time, mel)

        targets = torch.tensor([self.label_map.get(c, self.blank_idx) for c in sample.text], dtype=torch.long)

        return {
            "features": mel,  # (time, mel)
            "targets": targets,
            "utt_id": sample.path.stem,
            "text": sample.text,
        }


def collate_batch(
    batch: List[Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Pad variable-length inputs and targets for CTC training."""
    feature_lengths = torch.tensor([item["features"].shape[0] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([len(item["targets"]) for item in batch], dtype=torch.long)

    max_feat_len = int(feature_lengths.max())
    max_target_len = int(target_lengths.max())
    feat_dim = batch[0]["features"].shape[1]

    padded_feats = torch.zeros(len(batch), max_feat_len, feat_dim)
    pad_value = 0  # CTC ignores padding beyond target_lengths
    padded_targets = torch.full((len(batch), max_target_len), fill_value=pad_value, dtype=torch.long)

    utt_ids: List[str] = []
    texts: List[str] = []
    for i, item in enumerate(batch):
        t_len = item["features"].shape[0]
        padded_feats[i, :t_len] = item["features"]

        y_len = len(item["targets"])
        padded_targets[i, :y_len] = item["targets"]
        utt_ids.append(item["utt_id"])
        texts.append(item["text"])

    return padded_feats, feature_lengths, padded_targets, target_lengths, utt_ids, texts
