# audio2decodeMorse

Deep learning scaffold for decoding audible Morse code into text on macOS with GPU acceleration (MPS). The project is set up for training a CTC acoustic model on log-mel spectrograms generated from audio recordings containing Morse signals.

## Setup
- Create a virtual environment and install dependencies:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- PyTorch wheels for Apple Silicon/Intel macOS include MPS support. Verify the device at runtime (the training script does this automatically):
  ```python
  import torch; print(torch.backends.mps.is_available())
  ```
- Run scripts from the repo root with `PYTHONPATH=src` (e.g., `PYTHONPATH=src python3 -m audio2morse.training.train ...`).
- In VS Code you can use the integrated terminal to run the same commands, or add a Run/Debug configuration that executes `python3 -m audio2morse.training.train --config config/default.yaml` with `PYTHONPATH=src`.

## Data format
- Use a JSONL manifest with one sample per line:
  ```json
  {"audio_filepath": "data/audio/example.wav", "text": "SOS"}
  ```
- Optional metadata columns (useful for analysis) are supported and passed through manifests: `freq_hz`, `wpm`, `amplitude`. Synthetic data generation writes them automatically.
- Audio should be single-channel WAV. The loader will resample to the configured `sample_rate` (default 16000 Hz).
- Keep text uppercase A-Z, digits, basic punctuation, and spaces to match the default vocabulary.
- To build your own dataset:
  1. Record or generate clean Morse audio (e.g., 600–800 Hz tone, moderate speed) as mono WAV at 16 kHz.
  2. Write the corresponding plaintext (uppercase) for each file.
  3. Create `data/manifests/train.jsonl` and `data/manifests/val.jsonl` with lines like the example above.
  4. Ensure audio paths are correct relative to the repo root.
- There isn’t a widely used public Morse-audio benchmark; starting with your own recordings or synthetic tone generation works well to bootstrap a model.

## Visualize a WAV
- Plot and save a waveform for inspection:
  ```bash
  # one-time setup
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

  # generate a PNG waveform plot for your file
  PYTHONPATH=src python3 -m audio2morse.data.visualize_wav \
    --audio data/your_file.wav \
    --save outputs/your_file_waveform.png
  ```
- If you prefer to see it on-screen instead of saving, omit `--save` (works well in VS Code’s integrated terminal):
  ```bash
  PYTHONPATH=src python3 -m audio2morse.data.visualize_wav --audio data/your_file.wav
  ```
- The script converts stereo to mono automatically for plotting.

## Generate synthetic Morse data
- Prepare a text file with one message per line (uppercase recommended).
- Generate tone-based WAVs and a manifest (append mode) with:
  ```bash
  PYTHONPATH=src python3 -m audio2morse.data.generate_synthetic_morse \
    --input texts.txt \
    --out-dir data/audio \
    --manifest data/manifests/train.jsonl \
    --sample-rate 16000 \
    --wpm-min 18 --wpm-max 25 \
    --freq-min 500 --freq-max 900 \
    --amp-min 0.2 --amp-max 0.4
  ```
- This writes `synthetic_*.wav` files and appends entries to the JSONL manifest.
- Create a separate `texts_val.txt` if you want a validation set and point the manifest to `data/manifests/val.jsonl`.
- Flags `--wpm`, `--freq`, and `--amp` can fix those values; otherwise per-sample values are uniformly sampled from the min/max ranges to add variability.

## Training
- Edit `config/default.yaml` to point to your train/validation manifests and tweak hyperparameters.
- Run training:
  ```bash
  PYTHONPATH=src python3 -m audio2morse.training.train --config config/default.yaml
  ```
- Checkpoints are saved in `outputs/` by default (configurable).

## Inference
- After training, decode an audio file with greedy decoding:
  ```bash
  PYTHONPATH=src python3 -m audio2morse.inference.greedy_decode \
    --checkpoint outputs/best.pt \
    --audio data/audio/example.wav
  ```
- The script prints the predicted text.

## Project layout
- `config/` – YAML configs for experiments.
- `src/audio2morse/data/` – dataset, feature pipeline, vocabulary utilities.
- `src/audio2morse/models/` – acoustic model definition.
- `src/audio2morse/training/` – training loop entrypoint.
- `src/audio2morse/inference/` – inference helpers and CLI.
- `src/audio2morse/data/morse_map.py` – international Morse code dot/dash lookup tables.

## Model architecture
- Front-end: log-mel spectrograms (64 bins by default) with frame length/step 25/10 ms, mono audio resampled to 16 kHz.
- Encoder: 3 convolutional blocks (channels 32 → 64 → 128) with ReLU, dropout, and layer norm to denoise and downsample in frequency.
- Temporal modeling: 3-layer bidirectional LSTM (hidden size 256, dropout 0.2) to capture dot/dash timing.
- Head: linear projection to vocab logits; CTC loss with a `<BLANK>` token (see `src/audio2morse/models/ctc_model.py`).
- Configurable in `config/default.yaml` (`model` and `data` sections). You can change channels, RNN depth/width, and dropout there.

## Notes on CTC
- The model trains with Connectionist Temporal Classification (CTC), which handles variable-length audio → text without pre-aligned labels by using a frame-level “blank” token and collapsing repeats/blanks at decode time.
- Greedy decoding in `inference/greedy_decode.py` simply takes the argmax per frame and collapses repeats/blanks; beam search could improve accuracy but isn’t included here for brevity.

## Analysis utilities
- After you’ve run `analyses/add_predictions.py` to enrich manifests with inference text and loss, you can visualize loss vs partition/frequency/WPM/amplitude and fit a simple linear model:
  ```bash
  Rscript analyses/analyze_preds.R \
    --input analyses/combined_with_preds.csv \
    --out-dir analyses/figures
  ```
  The script installs required R packages if missing and writes plots plus `linear_model_summary.txt` to the output directory.

## License
This project is licensed under the MIT License (see `LICENSE`).
