#!/usr/bin/env python3
"""
Extension: Stylistic Paraphrase Generation (Modern English -> Shakespearean).

We reuse the SonnetGPT model (full-sequence logits + top-p generate +
weight-tied output projection) and fine-tune it with teacher forcing and
*source-side loss masking*: only the Shakespearean target tokens contribute to
the loss, the "Modern: ... \nShakespearean: " prompt is masked out.

Running:
  python prepare_style_data.py                          # once, downloads data
  python stylistic_paraphrase.py --use_gpu --epochs 5   # train + sample + chrF

See STYLISTIC_PARAPHRASE.md for the full design.
"""

import argparse
import random
import re
import torch

import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import StyleParaphraseDataset, load_style_data, make_style_prompt
from sonnet_generation import SonnetGPT, add_arguments, save_model
from optimizer import AdamW
from utils import get_device

TQDM_DISABLE = False


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def compute_masked_loss(model, b_ids, b_mask, prompt_lens):
  """Teacher-forcing cross-entropy over the TARGET tokens only."""
  logits = model(b_ids, b_mask)[:, :-1].contiguous()        # [B, T-1, V]
  labels = b_ids[:, 1:].clone()                             # [B, T-1]
  # Mask the source/prompt region: label index i predicts token i+1, so the
  # first (pl-1) label positions belong to the prompt.
  for i, pl in enumerate(prompt_lens):
    labels[i, :max(pl - 1, 0)] = -100
  labels[b_mask[:, 1:] == 0] = -100                         # padding -> no loss
  V = logits.size(-1)
  loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                         ignore_index=-100, reduction='mean')
  return loss


@torch.no_grad()
def eval_loss(model, dataloader, device):
  model.eval()
  total, n = 0.0, 0
  for batch in tqdm(dataloader, desc='dev', disable=TQDM_DISABLE):
    b_ids = batch['token_ids'].to(device)
    b_mask = batch['attention_mask'].to(device)
    loss = compute_masked_loss(model, b_ids, b_mask, batch['prompt_lens'])
    total += loss.item()
    n += 1
  return total / max(n, 1)


def _extract_target(full_text):
  """Keep only the Shakespearean rewrite: everything after the last
  'Shakespearean:' tag, up to the first newline. (SonnetGPT.generate strips the
  first few decoded chars, so we anchor on the target tag, not the full prompt.)
  Also strip a leading run of repeated symbol/punctuation tokens that GPT-2 tends
  to emit right after a space-terminated prompt."""
  out = full_text
  if 'Shakespearean:' in out:
    out = out.rsplit('Shakespearean:', 1)[1]
  out = out.split('\n', 1)[0].strip()
  out = re.sub(r'^[^A-Za-z0-9\'"]+', '', out).strip()  # drop leading ~~~~ / ---- / ____
  return out


@torch.no_grad()
def generate_for(model, modern_list, device, temperature, top_p, max_length=64):
  """Generate a Shakespearean rewrite for each modern sentence (batch=1)."""
  model.eval()
  preds = []
  for modern in tqdm(modern_list, desc='generate', disable=TQDM_DISABLE):
    prompt = make_style_prompt(modern)
    enc = model.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(device)
    _, full = model.generate(enc['input_ids'], temperature=temperature,
                             top_p=top_p, max_length=max_length)
    preds.append(_extract_target(full))
  return preds


def chrf_score(hyps, refs):
  from sacrebleu.metrics import CHRF
  return float(CHRF().corpus_score(hyps, [refs]).score)


def train(args):
  device = get_device(args.use_gpu)

  train_pairs = load_style_data('train', args.data_dir)
  dev_pairs = load_style_data('valid', args.data_dir)

  if args.smoke_limit:
    train_pairs = train_pairs[:args.smoke_limit]
    dev_pairs = dev_pairs[:max(args.smoke_limit // 4, 8)]
    print(f"[SMOKE] {len(train_pairs)} train / {len(dev_pairs)} dev")

  train_ds = StyleParaphraseDataset(train_pairs, args, max_length=args.max_length)
  dev_ds = StyleParaphraseDataset(dev_pairs, args, max_length=args.max_length)
  train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size,
                        collate_fn=train_ds.collate_fn)
  dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=args.batch_size,
                      collate_fn=dev_ds.collate_fn)

  args = add_arguments(args)
  model = SonnetGPT(args).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr)

  best_dev = float('inf')
  for epoch in range(args.epochs):
    model.train()
    train_loss, num_batches = 0.0, 0
    for batch in tqdm(train_dl, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      optimizer.zero_grad()
      loss = compute_masked_loss(model, b_ids, b_mask, batch['prompt_lens'])
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      num_batches += 1
      if device.type == 'mps' and num_batches % 50 == 0:
        torch.mps.empty_cache()

    train_loss /= max(num_batches, 1)
    dev_loss = eval_loss(model, dev_dl, device)
    dev_ppl = float(np.exp(min(dev_loss, 20)))
    print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, "
          f"dev loss :: {dev_loss:.3f}, dev ppl :: {dev_ppl:.2f}")

    if dev_loss < best_dev:
      best_dev = dev_loss
      save_model(model, optimizer, args, args.filepath)

    # Qualitative samples each epoch.
    print('--- sample rewrites ---')
    samples = [m for (m, _) in dev_pairs[:5]]
    for modern, pred in zip(samples, generate_for(model, samples, device,
                                                  args.temperature, args.top_p)):
      print(f'  modern     : {modern}')
      print(f'  shakespeare: {pred}\n')


@torch.no_grad()
def evaluate(args):
  """Load best checkpoint, generate on test, report chrF, and compare against
  the zero-shot (non-fine-tuned) GPT-2 baseline."""
  device = get_device(args.use_gpu)
  test_pairs = load_style_data('test', args.data_dir)
  if args.smoke_limit:
    test_pairs = test_pairs[:args.smoke_limit]
  modern = [m for (m, _) in test_pairs]
  gold = [s for (_, s) in test_pairs]

  # Fine-tuned model.
  saved = torch.load(args.filepath, weights_only=False)
  ft = SonnetGPT(saved['args']).to(device)
  ft.load_state_dict(saved['model'])
  ft_preds = generate_for(ft, modern, device, args.temperature, args.top_p)
  ft_chrf = chrf_score(ft_preds, gold)

  # Zero-shot pretrained baseline (no fine-tuning).
  base_args = add_arguments(argparse.Namespace(model_size=args.model_size))
  base = SonnetGPT(base_args).to(device)
  base_preds = generate_for(base, modern, device, args.temperature, args.top_p)
  base_chrf = chrf_score(base_preds, gold)

  with open(args.style_out, 'w', encoding='utf-8') as f:
    for m, g, p in zip(modern, gold, ft_preds):
      f.write(f"Modern: {m}\nGold: {g}\nPred: {p}\n\n")

  print(f"\n=== chrF (test, n={len(modern)}) ===")
  print(f"  zero-shot GPT-2 : {base_chrf:.2f}")
  print(f"  fine-tuned      : {ft_chrf:.2f}")
  print(f"  improvement     : {ft_chrf - base_chrf:+.2f}")
  print(f"predictions written to {args.style_out}")
  return {'zero_shot_chrf': base_chrf, 'fine_tuned_chrf': ft_chrf}


def get_args():
  p = argparse.ArgumentParser()
  p.add_argument("--data_dir", type=str, default="data/shakespeare_modern")
  p.add_argument("--style_out", type=str, default="predictions/style-test-output.txt")
  p.add_argument("--seed", type=int, default=11711)
  p.add_argument("--epochs", type=int, default=5)
  p.add_argument("--use_gpu", action='store_true')
  p.add_argument("--batch_size", type=int, default=16)
  p.add_argument("--lr", type=float, default=1e-5)
  p.add_argument("--max_length", type=int, default=128)
  p.add_argument("--temperature", type=float, default=0.7)
  p.add_argument("--top_p", type=float, default=0.9)
  p.add_argument("--smoke_limit", type=int, default=0,
                 help="if >0, limit dataset sizes for a fast smoke test")
  p.add_argument("--model_size", type=str, default='gpt2',
                 choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
  return p.parse_args()


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-style.pt'
  seed_everything(args.seed)
  train(args)
  evaluate(args)
