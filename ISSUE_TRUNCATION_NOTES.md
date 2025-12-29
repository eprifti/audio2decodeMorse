## Long-phrase truncation in CTC decoding

Observations:
- Greedy CTC decoding truncates long phrases; later characters are often missing. Mean abs length diff on `bilstm_clean_large_gpu0` val/test is ~24 chars.
- Current best decoder (`bilstm_clean_gpu0_regen`) still greedy; no beam search/LM.
- Downsample may be too aggressive on some models; CTC blanks dominate tail.

Suspected causes:
- Greedy decoding favors blanks; no length penalty or LM to keep hypotheses alive.
- Time downsampling reduces T' so long labels cannot fully align.
- Training skewed toward shorter utterances; over-blank bias or label smoothing.

Suggested fixes:
1) Add beam search with length normalization and optional char LM; small beam (10–20) should help.
2) Reduce downsample_time (fewer pooling/pre_subsample off) to give CTC more frames.
3) Decode with slight logit temperature < 1 or lower label smoothing to counter blank dominance.
4) Rebalance training toward longer sequences or ensure T'/label_len > 2–3 for longest targets.
5) If count head available, rescore beams with count prior to match predicted length.

Next steps when resuming:
- Implement a lightweight beam search decoder and evaluate on long phrases (e.g., subset of `bilstm_clean_large_gpu0` and `bilstm_clean_gpu0_regen`).
- Consider a bucketed count classifier to stabilize counts if count-first flow is revisited.
