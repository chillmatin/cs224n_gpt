#!/usr/bin/env python3
"""
Two extra diagnostics for the sonnet model that the report references but that
require loading the trained checkpoint (so they are not in reproduce_metrics.py):

  1. Held-out dev *perplexity* of the fine-tuned model -- a second, non-chrF
     generation metric (addresses the chrF-only limitation).
  2. *distinct-1 / distinct-2 / rep-4* of the generated continuations at each
     decoding temperature -- a quantitative diversity/repetition measure that
     backs the "low temperature -> repetition" claim instead of one example.

Both reuse the committed checkpoint 9_10-1e-05-sonnet.pt (NO retraining).

    conda activate cs224n_dfp
    python analyze_sonnet.py --use_gpu
"""
import argparse

import torch
import torch.nn.functional as F

from sonnet_generation import SonnetGPT, seed_everything
from datasets import SonnetsDataset


def get_device(use_gpu):
  if use_gpu and torch.backends.mps.is_available():
    return torch.device("mps")
  if use_gpu and torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


@torch.no_grad()
def held_out_perplexity(model, device, gold_path):
  """Token-level perplexity of the model on the full held-out dev sonnets."""
  ds = SonnetsDataset(gold_path)
  total_loss, total_tokens = 0.0, 0
  for _, text in ds:
    enc = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    ids = enc["input_ids"]
    mask = torch.ones_like(ids)
    logits = model(ids, mask)              # [1, T, V]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), reduction="sum")
    total_loss += loss.item()
    total_tokens += shift_labels.numel()
  nll = total_loss / total_tokens
  return float(torch.exp(torch.tensor(nll))), total_tokens


def ngrams(tokens, n):
  return list(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else []


def corpus_distinct_n(list_of_token_seqs, n):
  """Corpus-level distinct-n: pool n-grams across ALL continuations, then
  unique / total. (Per-sonnet ratios are noisy because each continuation is short,
  so we pool.)"""
  all_grams = [g for seq in list_of_token_seqs for g in ngrams(seq, n)]
  if not all_grams:
    return float("nan")
  return len(set(all_grams)) / len(all_grams)


@torch.no_grad()
def diversity_by_temperature(model, device, seed_path, temps, top_p, seed):
  ds = SonnetsDataset(seed_path)
  rows = []
  for t in temps:
    seed_everything(seed)  # same seed AND same conditioning (same 12 seeds) per temp
    seqs = []
    for _, prompt in ds:
      enc = model.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(device)
      prompt_len = enc["input_ids"].shape[1]
      out = model.generate(enc["input_ids"], temperature=t, top_p=top_p)[0][0]
      cont = out[prompt_len:].cpu().tolist()          # generated continuation only
      seqs.append(model.tokenizer.convert_ids_to_tokens(cont))
    d1 = corpus_distinct_n(seqs, 1)
    d2 = corpus_distinct_n(seqs, 2)
    r4 = 1.0 - corpus_distinct_n(seqs, 4)             # corpus-level rep-4
    n_tok = sum(len(s) for s in seqs)
    rows.append((t, d1, d2, r4, n_tok))
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--use_gpu", action="store_true")
  p.add_argument("--checkpoint", default="9_10-1e-05-sonnet.pt")
  p.add_argument("--gold_path", default="data/TRUE_sonnets_held_out_dev.txt")
  p.add_argument("--seed_path", default="data/sonnets_held_out_dev.txt")
  p.add_argument("--top_p", type=float, default=0.9)
  p.add_argument("--seed", type=int, default=11711)
  p.add_argument("--no_diversity", action="store_true",
                 help="skip the (slow) generation sweep; perplexities only")
  args = p.parse_args()

  device = get_device(args.use_gpu)
  print(f"device: {device}")
  saved = torch.load(args.checkpoint, weights_only=False)
  model = SonnetGPT(saved["args"]).to(device)
  model.load_state_dict(saved["model"])
  model.eval()

  ppl, ntok = held_out_perplexity(model, device, args.gold_path)
  # Zero-shot (pretrained, no sonnet fine-tuning) baseline for context: a SonnetGPT
  # built from the same args but WITHOUT loading our checkpoint is just pretrained
  # GPT-2. Perplexity is only interpretable relative to such a reference.
  base = SonnetGPT(saved["args"]).to(device)
  base.eval()
  base_ppl, _ = held_out_perplexity(base, device, args.gold_path)
  print(f"\n== Held-out dev perplexity (lower is better; over {ntok} target tokens) ==")
  print(f"  zero-shot pretrained GPT-2 = {base_ppl:.2f}")
  print(f"  fine-tuned sonnet model    = {ppl:.2f}")

  if args.no_diversity:
    return
  print(f"\n== Diversity / repetition by temperature (top_p={args.top_p}, corpus-level over 12 dev sonnets) ==")
  print(f"  {'temp':>5} {'distinct-1':>11} {'distinct-2':>11} {'rep-4':>8} {'gen_tokens':>11}")
  for t, d1, d2, r4, n_tok in diversity_by_temperature(
      model, device, args.seed_path, [0.7, 0.9, 1.0, 1.2], args.top_p, args.seed):
    print(f"  {t:>5} {d1:>11.3f} {d2:>11.3f} {r4:>8.3f} {n_tok:>11}")


if __name__ == "__main__":
  main()
