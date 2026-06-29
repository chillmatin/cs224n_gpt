# Reproducing the report numbers

Every statistic in `report/main.tex` is listed below with the exact command that
produces it. All commands run from the repository root with the course conda
environment active:

```bash
conda activate cs224n_dfp
unalias python 2>/dev/null          # a global alias can shadow the env python
export PYTORCH_ENABLE_MPS_FALLBACK=1 # Apple Silicon / MPS backend
```

There are two ways to reproduce a number:

* **Fast (seconds, CPU, no retraining).** Re-score the saved prediction /
  generation artifacts in `predictions/` against the gold labels in `data/`:

  ```bash
  python reproduce_metrics.py
  ```

  This prints Tables 3–6 of the report and matches the paper exactly
  (SST-5 0.461/0.513, CFIMDB 0.857/0.967, paraphrase 0.889, sonnet dev chrF
  41.49, extension chrF 35.83).

* **Full (re-train / re-generate from scratch).** Run the task scripts below.
  These were the runs that produced the artifacts above; on an Apple M4 Pro (MPS)
  they take from a few minutes (extension, sonnet) to a few hours (paraphrase).

---

## Part 0 — implementation sanity checks
```bash
python optimizer_test.py     # -> "Optimizer test passed!"
python sanity_check.py       # -> "Your GPT2 implementation is correct!"
```

## Table 3 — Sentiment classification (dev acc / macro-F1)
Train + predict (writes `predictions/{mode}-{sst,cfimdb}-dev-out.csv`):
```bash
python classifier.py --fine-tune-mode last-linear-layer --use_gpu --lr 1e-3
python classifier.py --fine-tune-mode full-model        --use_gpu --lr 1e-5
# CFIMDB full-model used the MPS-safe config: batch 4, max_len 256, 5 epochs
```
Score the saved dev predictions against gold:
```bash
python reproduce_metrics.py   # see the "Table 3" block
```

## Table 4 — Paraphrase detection (Quora dev acc)
Train (3 epochs, cloze head, max_len 128 + left truncation, batch 16):
```bash
python paraphrase_detection.py --use_gpu --epochs 3 --batch_size 16 --lr 1e-5
# writes predictions/para-dev-output.csv  (tokens 8505="yes", 3919="no")
```
Score:
```bash
python reproduce_metrics.py   # "Table 4" block -> 0.889  (epochs 0/1/2: 0.863/0.879/0.889)
```

## Table 5 — Sonnet generation (dev chrF, temperature ablation)
Train once (10 epochs -> checkpoint `9_10-1e-05-sonnet.pt`):
```bash
python sonnet_generation.py --use_gpu --epochs 10 --lr 1e-5 --top_p 0.9
```
Sweep decoding temperature on the held-out **dev** set (no retraining; regenerates
from the checkpoint and scores against `data/TRUE_sonnets_held_out_dev.txt`):
```bash
for T in 0.7 0.9 1.0 1.2; do
  python eval_sonnet_dev.py --use_gpu --temperature $T --top_p 0.9
done
# dev chrF: 0.7 -> 39.01, 0.9 -> 40.63, 1.0 -> 41.49 (selected), 1.2 -> 41.12
```
Fast re-score of the saved tau=1.0 generation:
```bash
python reproduce_metrics.py   # "Table 5" block -> 41.49
```

## Tables 6 & 7 — Extension: stylistic paraphrase (test chrF + prompt ablation)
Download/prepare the Jhamtani parallel data, then train (5 epochs):
```bash
python prepare_style_data.py
python stylistic_paraphrase.py --use_gpu --epochs 5 --batch_size 16 --lr 1e-5
# prints: zero-shot 13.85 -> fine-tuned 35.83 (+21.97); writes predictions/style-test-output.txt
```
Fast re-score of the saved fine-tuned outputs:
```bash
python reproduce_metrics.py   # "Table 6" block -> 35.83
```
Prompt-design ablation (Table 7): the spaced-tag variant (`"Shakespearean: "`,
29.52 chrF) vs. the final unspaced tag (`"Shakespearean:"` + `" {target}"`,
35.83 chrF) is toggled by the prompt construction in `datasets.py` /
`stylistic_paraphrase.py`; re-run the training command above with each variant to
reproduce the two rows.

## Analysis diagnostics (Analysis section)

**Sonnet held-out perplexity + per-temperature diversity** (no retraining; loads
the saved checkpoint `9_10-1e-05-sonnet.pt`):
```bash
python analyze_sonnet.py --use_gpu
# held-out dev perplexity (lower is better): zero-shot GPT-2 = 99.1 -> fine-tuned = 59.0
# corpus-level rep-4 by temperature: 0.7->0.170, 0.9->0.038, 1.0->0.008, 1.2->0.002
# distinct-2: 0.7->0.506, 0.9->0.712, 1.0->0.811, 1.2->0.884
```

**CFIMDB batch-size hypothesis test** (controlled gradient-accumulation experiment;
writes only to a scratch dir, leaves `cfimdb-classifier.pt` untouched):
```bash
python run_cfimdb_batch.py all
# effective batch  4 (accum_steps=1): dev acc = 0.9673  <- reproduces the original 0.967
#                                                          (sanity check for the grad-accum refactor)
# effective batch 32 (accum_steps=8): dev acc = 0.9592  <- larger batch does NOT close the gap
# -> batch-size hypothesis not supported; residual gap to 0.976 unexplained (single-seed)
```

---

## Notes
* `reproduce_metrics.py` loads no model and trains nothing — it only re-applies the
  scoring functions (scikit-learn accuracy/F1 and the course `evaluation.test_sonnet`
  / sacrebleu `CHRF`) to the committed artifacts, so the reported numbers are
  independently checkable in seconds.
* Test-set scores are intentionally **not** listed here: the official test labels
  are hidden, so we report dev / held-out numbers throughout.
