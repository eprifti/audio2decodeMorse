# Self-Supervised Envelope Experiments (1/3/7 prior)

This note summarizes the envelope-based self-supervision experiments, mask sweeps, beam decode, and a supervised fine-tune starting from the envelope-pretrained checkpoint.

## Setups and commands
- Self-supervised pretrain on `data/datasets/large_baseline/*` (10 epochs, batch 64):
  ```
  PYTHONPATH=src .venv/bin/python analyses/train_self_supervised_envelope.py \
    --train-manifest data/datasets/large_baseline/manifests/train.jsonl \
    --val-manifest   data/datasets/large_baseline/manifests/val.jsonl \
    --smooth-ms 10 --threshold-ratio 0.3 --mask-weight 0.5 \
    --epochs 10 --batch-size 64 --out-dir outputs/self_supervised_envelope_large
  ```
  Final losses (epoch 10): train_loss ≈ 0.0972, val_loss ≈ 0.0757.

- Eval script (adds beam search and sample dumps):
  ```
  PYTHONPATH=src .venv/bin/python analyses/eval_self_supervised_envelope.py \
    --checkpoint outputs/self_supervised_envelope_large/best.pt \
    --manifest data/datasets/large_baseline/manifests/val.jsonl \
    --batch-size 64 --beam-size 3 --device cuda \
    --max-samples 200 --print-samples 5
  ```
  Greedy baseline (full val earlier): CER ~0.732, exact ~2.6%.  
  Beam=3 on 200 samples: CER ≈ 0.7085, exact ≈ 2.0%; sample decodes show truncation/garbling.

- Mask-weight sweep (1 epoch, batch 128, full dataset):
  - `mask_weight=0.3`: train_loss=2.9761 val_loss=1.9827 (`outputs/self_supervised_envelope_large_mask03`)
  - `mask_weight=0.5`: val_loss≈1.9955 (baseline run)
  - `mask_weight=0.7`: train_loss=3.3899 val_loss=2.5139 (`outputs/self_supervised_envelope_large_mask07`)
  => Lower mask weight (0.3–0.5) is better; 0.7 degrades.

- Supervised fine-tune from envelope checkpoint (1 epoch, batch 128):
  ```
  PYTHONPATH=src .venv/bin/python analyses/finetune_supervised_from_ckpt.py \
    --checkpoint outputs/self_supervised_envelope_large/best.pt \
    --train-manifest data/datasets/large_baseline/manifests/train.jsonl \
    --val-manifest   data/datasets/large_baseline/manifests/val.jsonl \
    --epochs 1 --batch-size 128 --device cuda \
    --out-dir outputs/finetune_supervised_from_envelope
  ```
  Result: train_loss=0.0955, val_loss=0.0941.

- Eval of the self-supervised model (greedy, 1k-sample subset):
  ```
  PYTHONPATH=src .venv/bin/python analyses/eval_self_supervised_envelope.py \
    --checkpoint outputs/self_supervised_envelope_large/best.pt \
    --manifest data/datasets/large_baseline/manifests/val.jsonl \
    --batch-size 64 --beam-size 1 --device cuda \
    --max-samples 1000 --print-samples 3
  ```
  CER ≈ 0.7185, exact ≈ 33.2%. Outputs `combined_with_preds.csv` and `loss_trace.csv` in the checkpoint dir.

- Eval of the fine-tuned model (greedy, refs filtered to A–Z0–9 space, 1k samples):
  ```
  PYTHONPATH=src .venv/bin/python analyses/eval_self_supervised_envelope.py \
    --checkpoint outputs/finetune_supervised_from_envelope/best.pt \
    --manifest data/datasets/large_baseline/manifests/val.jsonl \
    --batch-size 64 --beam-size 1 --device cuda \
    --max-samples 1000 --print-samples 3
  ```
  CER ≈ 0.777, exact-match ≈ 36.6%. Many short phrases decode perfectly; long/complex ones still hurt CER.

## Takeaways
- Envelope self-supervision alone yields low training loss but high CER; beam search helps only slightly.
- Auxiliary mask weight should stay around 0.3–0.5; higher weights hurt.
- A single supervised fine-tune epoch from the envelope checkpoint dramatically lowers CTC loss, boosts exact-match on short phrases, but CER remains high on the full val set—needs more supervised epochs/decoding tweaks.
- Decoding quality is still the bottleneck; length drift and truncation persist on longer strings.

## Suggested next steps
1) Run 3–5 supervised fine-tune epochs from `outputs/finetune_supervised_from_envelope/best.pt`, then beam-evaluate (beam 5–10) on val/test.
2) Add punctuation tokens to the alphabet if the manifests include them; current eval filters to A–Z0–9+space, inflating CER when refs carry punctuation.
3) Experiment with larger batch (memory allows) and label smoothing/CTC dropout; keep mask weight ≤0.5 if redoing self-supervised pretrain.
4) Inspect a handful of long-form decodes after fine-tune to pinpoint failure modes (length drift vs substitutions) and adjust decoding constraints accordingly.
