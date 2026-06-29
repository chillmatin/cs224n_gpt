#!/usr/bin/env python3
"""
Reproduce every headline number in the report (report/main.tex) directly from the
saved prediction / generation artifacts in predictions/ and the gold labels in
data/. No model is loaded and no training is run, so this completes in seconds on
CPU and verifies that the reported scores are real and reproducible.

Run inside the course conda env (it needs scikit-learn and sacrebleu):

    conda activate cs224n_dfp
    python reproduce_metrics.py

Mapping of what this prints to the report tables (numbers as they appear in
report/main.tex):
    Table 3 (sentiment dev acc / F1)  <- predictions/*-{sst,cfimdb}-dev-out.csv
    Table 4 (paraphrase dev acc)      <- predictions/para-dev-output.csv
    Table 5 (sonnet dev chrF, tau=1.0)<- predictions/generated_sonnets_dev.txt
    Table 6 (extension test chrF, FT) <- predictions/style-test-output.txt

The full ablation sweeps (sonnet temperatures 0.7/0.9/1.2, the extension
zero-shot 13.85 baseline, and the spaced-prompt 29.52 run from Table 7) require
re-running the training/generation scripts; those commands are listed in
REPRODUCE.md.
"""
import csv

from sklearn.metrics import accuracy_score, f1_score


# ----------------------------------------------------------------------------- #
# Table 3 -- sentiment classification (dev accuracy and macro-F1)
# ----------------------------------------------------------------------------- #
def load_sentiment_gold(path):
    gold = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            gold[row['id'].strip()] = int(row['sentiment'].strip())
    return gold


def load_sentiment_pred(path):
    pred = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, label = line.rsplit(',', 1)
            pred[sid.strip()] = int(label.strip())
    return pred


def sentiment_table():
    print("== Table 3: Sentiment classification (dev) ==")
    jobs = [
        ("SST-5  last-linear", "predictions/last-linear-layer-sst-dev-out.csv", "data/ids-sst-dev.csv"),
        ("SST-5  full-model ", "predictions/full-model-sst-dev-out.csv",        "data/ids-sst-dev.csv"),
        ("CFIMDB last-linear", "predictions/last-linear-layer-cfimdb-dev-out.csv", "data/ids-cfimdb-dev.csv"),
        ("CFIMDB full-model ", "predictions/full-model-cfimdb-dev-out.csv",       "data/ids-cfimdb-dev.csv"),
    ]
    for name, pred_path, gold_path in jobs:
        gold = load_sentiment_gold(gold_path)
        pred = load_sentiment_pred(pred_path)
        ids = sorted(set(gold) & set(pred))
        yt = [gold[i] for i in ids]
        yp = [pred[i] for i in ids]
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average='macro')
        print(f"  {name}: acc = {acc:.4f}  macro-F1 = {f1:.4f}  (n={len(ids)})")
    print()


# ----------------------------------------------------------------------------- #
# Table 4 -- cloze-style paraphrase detection (Quora dev accuracy)
# The cloze head emits BPE token 8505 ("yes" -> paraphrase=1) or 3919 ("no" -> 0).
# ----------------------------------------------------------------------------- #
YES_TOKEN, NO_TOKEN = 8505, 3919


def paraphrase_table():
    print("== Table 4: Paraphrase detection (Quora dev) ==")
    gold = {}
    with open("data/quora-dev.csv") as f:
        for row in csv.DictReader(f, delimiter='\t'):
            gold[row['id'].strip()] = int(float(row['is_duplicate']))
    pred = {}
    with open("predictions/para-dev-output.csv") as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, tok = line.rsplit(',', 1)
            tok = int(tok.strip())
            pred[sid.strip()] = 1 if tok == YES_TOKEN else 0
    ids = sorted(set(gold) & set(pred))
    yt = [gold[i] for i in ids]
    yp = [pred[i] for i in ids]
    print(f"  GPT-2 small, full FT, cloze (final): acc = {accuracy_score(yt, yp):.4f}  (n={len(ids)})")
    print()


# ----------------------------------------------------------------------------- #
# Table 5 -- sonnet generation dev chrF (selected temperature tau=1.0)
# Uses the exact same scorer as the course harness (evaluation.test_sonnet).
# ----------------------------------------------------------------------------- #
def sonnet_table():
    print("== Table 5: Sonnet generation (dev chrF, tau=1.0) ==")
    try:
        from evaluation import test_sonnet
        score = test_sonnet(
            test_path="predictions/generated_sonnets_dev.txt",
            gold_path="data/TRUE_sonnets_held_out_dev.txt",
        )
        print(f"  dev chrF (top_p=0.9, temp=1.0) = {score:.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"  [skipped: {e}]")
    print()


# ----------------------------------------------------------------------------- #
# Table 6 -- extension: stylistic paraphrase test chrF (fine-tuned)
# Re-scores the saved Gold/Pred pairs with the same CHRF metric.
# ----------------------------------------------------------------------------- #
def extension_table():
    print("== Table 6: Stylistic paraphrase extension (test chrF, fine-tuned) ==")
    try:
        from sacrebleu.metrics import CHRF
        golds, preds = [], []
        with open("predictions/style-test-output.txt") as f:
            for line in f:
                if line.startswith("Gold: "):
                    golds.append(line[len("Gold: "):].rstrip("\n"))
                elif line.startswith("Pred: "):
                    preds.append(line[len("Pred: "):].rstrip("\n"))
        n = min(len(golds), len(preds))
        score = float(CHRF().corpus_score(preds[:n], [golds[:n]]).score)
        print(f"  fine-tuned (source-side masking) chrF = {score:.2f}  (n={n})")
    except Exception as e:  # noqa: BLE001
        print(f"  [skipped: {e}]")
    print()


if __name__ == "__main__":
    sentiment_table()
    paraphrase_table()
    sonnet_table()
    extension_table()
