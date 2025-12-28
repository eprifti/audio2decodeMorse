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
- Prepare `data/texts.txt` with one message per line (uppercase recommended).
- Fast multi-core generation with progress bar (splits train/val/test automatically):
  ```bash
  PYTHONPATH=src python3 -m audio2morse.data.generate_synthetic_morse --config config/generation.yaml
  ```
  - Uses all CPU cores by default (`num_workers: -1`), shows a single progress line, and writes WAVs to `data/datasets/simple_baseline/audio`.
  - Manifests land in `data/datasets/simple_baseline/manifests/{train,val,test}.jsonl` with metadata (`freq_hz`, `wpm`, `amplitude`, `text_len`).
  - Adjust `config/generation.yaml` to change ranges, sample counts, or target characters; CLI flags still override the config if provided.
- To over-sample troublesome symbols or phrases, use the `--target-chars` / `--target-samples` or `--target-words-file` options (see `generate_synthetic_morse.py` docstring).
- You can also drive generation from a YAML config (see `config/generation.yaml` for a full-coverage 10k example with broad speed/pitch/amp variation and alphabet oversampling):
  ```bash
  PYTHONPATH=src python3 -m audio2morse.data.generate_synthetic_morse \
    --config config/generation.yaml
  ```
  By default this reads `data/texts.txt`, which now contains a diverse set of phrases, punctuation, numbers, and operating phrases to cover the full alphabet, and splits into train/val/test manifests.
  You can still override any field on the CLI (e.g., `--num-samples 2000`); required args are satisfied by the config.
  To emphasize spaces/word structure further, you can inject random phrases from a word list:
  ```bash
  PYTHONPATH=src python3 -m audio2morse.data.generate_synthetic_morse \
    --config config/generation.yaml \
    --target-words-file data/texts.txt \
    --target-words-samples 500 \
    --target-words-min 2 \
    --target-words-max 6
  ```

## Training
- Edit `config/default.yaml` (or the baseline configs in `config/`) to point to your train/validation manifests and tweak hyperparameters.
- Run training (timestamped run name by default; or pass your own):
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src python3 -m audio2morse.training.train \
    --config config/baseline_small_cnn2_lstm128.yaml \
    --run-name baseline_small_cnn2_lstm128
  ```
- If you omit `--run-name`, a timestamped name like `run-20241227-153045` is created. Checkpoints and loss curves are saved under `outputs/<run-name>/` (root is configurable in `training.checkpoint_dir`). A copy of the config is stored as `config_used.yaml` in the run folder. Early stopping is disabled by default; training runs for the configured epochs.
- Default training uses SpecAugment (time/frequency masking) to improve robustness; adjust/disable in `data.augment.specaugment`.
- Additional waveform augments (random gain, light noise) are enabled by default in `data.augment.waveform`.
- Learning-rate schedule: ReduceLROnPlateau on validation loss is enabled (see `training.lr_scheduler`); weight decay set to 1e-4.

## One-shot pipeline
- Small LSTM baseline end-to-end:
  ```bash
  ./launch_run_small.sh            # or ./launch_run_small.sh my_run_name
  ```
- Clean biLSTM baseline end-to-end:
  ```bash
  ./launch_run_bilstm_clean.sh     # or ./launch_run_bilstm_clean.sh my_run_name
  ```
- Multi-task prototype (CTC + bit/gap heads):
  ```bash
  ./launch_run_multitask.sh        # or ./launch_run_multitask.sh my_run_name
  ```
- Multi-task counts prototype (CTC + bit/gap + counts/hist):
  ```bash
  ./launch_run_multitask_counts.sh # or ./launch_run_multitask_counts.sh my_run_name
  ```
- Transformer CTC prototype:
  ```bash
  ./launch_run_transformer.sh      # or ./launch_run_transformer.sh my_run_name
  ```
Both scripts regenerate data via `config/generation.yaml`, train with the chosen config, add predictions to train/val/test manifests, and run the R analysis. Outputs live under `outputs/<run-name>/`.

## Inference
- After training, decode an audio file with greedy decoding:
  ```bash
  PYTHONPATH=src python3 -m audio2morse.inference.greedy_decode \
    --checkpoint outputs/baseline_small_cnn2_lstm128/best.pt \
    --audio data/datasets/simple_baseline/audio/synthetic_00001.wav
  ```
- The script prints the predicted text.
- For potentially better accuracy, enable a small CTC beam search:
  ```bash
  PYTHONPATH=src python3 -m audio2morse.inference.greedy_decode \
    --checkpoint outputs/baseline_small_cnn2_lstm128/best.pt \
    --audio data/datasets/simple_baseline/audio/synthetic_00001.wav \
    --beam-size 5
  ```

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

### Prototype multi-task model
- `src/audio2morse/models/multitask_ctc.py` introduces a prototype with auxiliary heads:
  - Text CTC logits (as usual).
  - Bit head (dot/dash/blank) and gap head (none/char-gap/word-gap) to encourage better timing/segmentation.
- See `config/prototype_multitask.yaml` for suggested hyperparameters. Training integration would need extra supervision for the auxiliary heads (bit/gap labels derived from Morse timings).

### Multi-task with counts
- `src/audio2morse/models/multitask_ctc_counts.py` extends the multi-task model with:
  - Character count regression head (predicts total characters including spaces).
  - Character histogram head (bag-of-character counts excluding the blank token).
- Use `config/prototype_multitask_counts.yaml` and the launcher `launch_run_multitask_counts.sh` to try it. The training loop adds auxiliary MSE losses on counts and per-character counts in addition to the CTC loss.

### Transformer variant
- `src/audio2morse/models/transformer_ctc.py` replaces the LSTM with a Transformer encoder (CNN front-end + positional encoding + TransformerEncoder).
- Config: `config/transformer_small.yaml` and launcher `launch_run_transformer.sh`.

## Perspectives
- Temporal context matters for Morse (dot/dash duration and gaps). Besides LSTMs, you can try:
  - Temporal CNN/TCN stacks to model short-range timing without recurrence.
  - Light transformers or conformers with limited attention span for local context.
- Segmentation-aware training:
  - Supervise bit/gap heads (or the counts model) using labels derived from manifests (text length, character counts, inferred pauses) to improve boundary detection.
  - Evaluate per-frame bit/gap accuracy and boundary precision/recall on synthetic data where timing is known.

## Notes on CTC
- The model trains with Connectionist Temporal Classification (CTC), which handles variable-length audio → text without pre-aligned labels by using a frame-level “blank” token and collapsing repeats/blanks at decode time.
- Greedy decoding in `inference/greedy_decode.py` simply takes the argmax per frame and collapses repeats/blanks; beam search could improve accuracy but isn’t included here for brevity.

## Analysis utilities
- After you’ve run `analyses/add_predictions.py` (or `./launch_run.sh`) to enrich manifests with inference text and loss, you can visualize loss vs partition/frequency/WPM/amplitude and fit a simple linear model:
  ```bash
  Rscript analyses/analyze_preds.R --run-dir outputs/<run-name>
  ```
  The script installs required R packages if missing and writes plots plus `linear_model_summary.txt` to `outputs/<run-name>/`.

## License
This project is licensed under the MIT License (see `LICENSE`).
