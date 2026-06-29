#!/usr/bin/env python3
"""
Download the Jhamtani "No Fear Shakespeare" parallel corpus and save it as
sentence-aligned plain-text files for the stylistic paraphrase extension.

Source: github.com/harsh19/Shakespearizing-Modern-English  (data/*.nltktok)
Direction: modern (source)  ->  original / Shakespearean (target)

We download the raw .nltktok files with `requests` (no HuggingFace `datasets`
library — it is not installed and would in any case be shadowed by the local
datasets.py module). Output:

    data/shakespeare_modern/{train,valid,test}.modern.txt
    data/shakespeare_modern/{train,valid,test}.original.txt

Run once:  python prepare_style_data.py
"""

import os
import urllib.request

RAW_BASE = "https://raw.githubusercontent.com/harsh19/Shakespearizing-Modern-English/master/data"
OUT_DIR = "data/shakespeare_modern"

# split -> expected aligned pair count (verified live; used as a sanity check)
EXPECTED = {"train": 18395, "valid": 1218, "test": 1462}
SIDES = {"modern": "modern", "original": "original"}


def _detok(line):
  """The corpus is whitespace-tokenized (nltk). Lightly de-tokenize so the text
  reads naturally and tokenizes well with the GPT-2 BPE tokenizer."""
  s = line.strip()
  # glue punctuation back onto the preceding word
  for p in [" .", " ,", " ?", " !", " ;", " :", " '", " n't", " 's", " 're", " 've", " 'm", " 'll", " 'd"]:
    s = s.replace(p, p[1:])
  s = s.replace("`` ", '"').replace(" ''", '"').replace("``", '"').replace("''", '"')
  s = s.replace("( ", "(").replace(" )", ")")
  return " ".join(s.split())


def fetch(split, side):
  url = f"{RAW_BASE}/{split}.{side}.nltktok"
  print(f"  GET {url}")
  with urllib.request.urlopen(url, timeout=60) as resp:
    text = resp.read().decode("utf-8")
  lines = [ln for ln in text.splitlines() if ln.strip()]
  return [_detok(ln) for ln in lines]


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  for split, expected in EXPECTED.items():
    modern = fetch(split, "modern")
    original = fetch(split, "original")
    assert len(modern) == len(original), \
      f"{split}: modern={len(modern)} != original={len(original)} (alignment broken)"
    if len(modern) != expected:
      print(f"  [warn] {split}: got {len(modern)} pairs, expected {expected} "
            f"(corpus may have been updated upstream)")
    with open(os.path.join(OUT_DIR, f"{split}.modern.txt"), "w", encoding="utf-8") as f:
      f.write("\n".join(modern) + "\n")
    with open(os.path.join(OUT_DIR, f"{split}.original.txt"), "w", encoding="utf-8") as f:
      f.write("\n".join(original) + "\n")
    print(f"{split}: {len(modern)} aligned pairs -> {OUT_DIR}/{split}.{{modern,original}}.txt")

  print("\nDone. Example pair (train[0]):")
  with open(os.path.join(OUT_DIR, "train.modern.txt"), encoding="utf-8") as f:
    m0 = f.readline().strip()
  with open(os.path.join(OUT_DIR, "train.original.txt"), encoding="utf-8") as f:
    o0 = f.readline().strip()
  print(f"  modern     : {m0}")
  print(f"  shakespeare: {o0}")


if __name__ == "__main__":
  main()
