# CS 224N Default Final Project: Build GPT-2

Implementation of GPT-2 fine-tuned for sentiment classification, paraphrase detection, and sonnet generation.
Adapted for CENG534 DLNLP at IZTECH.

---

## Setup

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

```bash
conda env create -f env.yml
conda activate cs224n_dfp
```

**Apple Silicon (M-series Mac):** The install above works as-is. PyTorch uses the MPS backend automatically — no extra steps needed.

---

## Verify the implementation (Part 1)

Run both checks before training anything:

```bash
python optimizer_test.py
python sanity_check.py
```

Both should complete without errors.

---

## Training

All scripts must be run from the project root with the conda env activated.

### Option A: Run everything in one shot (recommended)

```bash
bash run_all_training.sh
```

This runs CFIMDB full-model training first, then paraphrase detection sequentially.
Logs are written to `logs_cfimdb.txt` and `logs_paraphrase.txt`.

### Option B: Run each task separately

**Sentiment classification — SST and CFIMDB (last-linear-layer mode):**
```bash
python classifier.py --use_gpu --fine-tune-mode last-linear-layer --epochs 10 --lr 1e-3
```

**Sentiment classification — SST (full-model mode):**
```bash
python classifier.py --use_gpu --fine-tune-mode full-model --epochs 10 --lr 1e-5
```

**Sentiment classification — CFIMDB (full-model mode only):**
```bash
python run_cfimdb_fullmodel.py
```

**Paraphrase detection:**
```bash
python run_paraphrase.py
```

**Sonnet generation:**
```bash
python sonnet_generation.py --use_gpu --epochs 10 --lr 1e-5
```

---

## Batch size notes (M4 Pro 24 GB)

The batch sizes below are the values actually used to produce the reported
numbers; the CFIMDB and paraphrase wrappers are tuned for 24 GB unified memory:

| Task | batch_size | Set in |
|------|-----------|--------|
| SST classifier (last-linear / full) | 8 | `classifier.py` default |
| CFIMDB full-model | 4 | `run_cfimdb_fullmodel.py` |
| Paraphrase detection | 16 | `run_paraphrase.py` |
| Sonnet generation | 8 | `sonnet_generation.py` default |

If you see an out-of-memory error, halve the `batch_size` in the relevant script.

---

## Generating submission files

After all training is complete, verify prediction files exist:

```bash
ls predictions/
```

Expected files:
- `last-linear-layer-sst-dev-out.csv`
- `last-linear-layer-sst-test-out.csv`
- `full-model-sst-dev-out.csv`
- `full-model-sst-test-out.csv`
- `last-linear-layer-cfimdb-dev-out.csv`
- `last-linear-layer-cfimdb-test-out.csv`
- `full-model-cfimdb-dev-out.csv`
- `full-model-cfimdb-test-out.csv`
- `para-dev-output.csv`
- `para-test-output.csv`
- `generated_sonnets.txt`

Then create the submission zip:

```bash
python prepare_submit.py
```

---

## Project structure

```
modules/attention.py       — Causal multi-head self-attention
modules/gpt2_layer.py      — GPT-2 transformer layer (pre-norm)
models/gpt2.py             — Full GPT-2 model, weight-tied output projection
optimizer.py               — AdamW with decoupled weight decay
classifier.py              — Sentiment classification (SST / CFIMDB)
paraphrase_detection.py    — Paraphrase detection (Quora dataset)
sonnet_generation.py       — Autoregressive sonnet generation
run_cfimdb_fullmodel.py    — CFIMDB full-model training script
run_paraphrase.py          — Paraphrase training script
run_all_training.sh        — Sequential runner for both scripts
```

---

## Acknowledgements

Adapted from the Stanford CS 224N default final project.
Parts of the code are from the [transformers](https://github.com/huggingface/transformers) library (Apache License 2.0).
