# Experiment Summary

Recent runs and their best validation metrics (lower is better for losses).

| Run | Model | Val metric | Notes |
| --- | --- | --- | --- |
| `transformer_tuned_two_gpu` | Transformer CTC (pre-sub-sample, label smoothing, warmup-cosine) | 3.81 CTC loss | Best transformer so far; trained on regenerated data ranges (wpm 15–30, freq 350–1000, amp 0.15–0.45). |
| `transformer_small_two_gpu` | Transformer small | ~4.31 CTC loss | Baseline small transformer. |
| `transformer_small_two_gpu_reg` | Transformer small + regularization | ~4.32 CTC loss | Similar to small baseline. |
| `transformer_small_four_gpu` | Transformer small | ~4.32 CTC loss | DP on 4 GPUs. |
| `transformer_tuned_ls_cos` | Transformer tuned | 4.44 best early, diverged later | Early overfit/divergence. |
| `transformer_tuned_ls_cos_v2` | Transformer tuned + broader data | 4.66 CTC loss | Uses widened generation ranges. |
| `baseline_small_two_gpu_v2` | CNN2 + LSTM128 | ~4.67 CTC loss | Legacy small LSTM baseline. |
| `bilstm_clean_gpu0_regen` | CNN3 + BiLSTM256 (CTC) | **1.20 CTC loss** | Best text-decoder to date; trained on regenerated data. |
| `bilstm_counts_gpu0_regen` | CTC + count/hist heads | 82.6 combined loss | Count aux dominated; text quality degraded. |
| `charcount_log_gpu0_regen` | Count-only (log target) | 0.078 val MSE (log-count) | Predicts only character counts; best checkpoint in `outputs/charcount_log_gpu0_regen/best.pt`. |

Artifacts of interest:
- Each run folder in `outputs/` contains `combined_with_preds.*`, `per_char_errors.csv`, `figures/` (R analysis), `loss_history.csv`, and `best.pt`.
- Count-only manifests now include `predicted_count`, `true_count`, and MSE (`loss`) per utterance.

Next steps suggested:
1. If focusing on count-first → decode-later: consider a bucketed count classifier (0–40) with cross-entropy to stabilize counts further, or continue with the log-regression if MAE is acceptable.
2. For decoding, the `bilstm_clean_gpu0_regen` checkpoint is the current leader; using its logits plus a light beam search could further reduce errors.
3. Transformers underperform relative to the tuned BiLSTM; retrain them with the latest data/augment and perhaps a smaller LR (2e-4) plus stronger dropout/specaugment for fair comparison.
