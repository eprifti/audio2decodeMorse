# Model runs

Checkpoints, configs, and logs are written under `outputs/<run-name>/` by the
training script. If you omit `--run-name`, a timestamped name like
`run-20241227-153045` is created automatically.

- `config_used.yaml` — copy of the config used for the run.
- `epoch_*.pt`, `best.pt` — checkpoints.
- `loss_history.csv`, `loss_curve.png` — training/validation loss traces.
- (Optionally) prediction/analysis outputs if you pass `--run-dir` to the
  analysis scripts.

To train with a specific name:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv311/bin/python \
  -m audio2morse.training.train \
  --config config/baseline_small_cnn2_lstm128.yaml \
  --run-name baseline_small_cnn2_lstm128
```

To decode with that run:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv311/bin/python \
  -m audio2morse.inference.greedy_decode \
  --checkpoint outputs/baseline_small_cnn2_lstm128/best.pt \
  --audio data/audio/example.wav \
  --beam-size 5
```

Alternate configs:
- `config/small_baseline.yaml`: smaller model (cnn_channels [32, 64], rnn_hidden_size 128, rnn_layers 2, dropout 0.1, unidirectional).
  Example run: `--config config/small_baseline.yaml --run-name baseline_small_cnn2_lstm128`.
