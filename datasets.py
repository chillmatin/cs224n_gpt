# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv

import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


def make_cloze_prompt(s1, s2):
  """Cloze-style paraphrase prompt.

  Train and test MUST use the identical template, otherwise the model is evaluated
  on a format it never saw during training (which silently hurts the test/leaderboard
  score while dev — built from the same template — still looks fine).
  """
  return f'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?\n'


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    # labels = torch.LongTensor([x[2] for x in all_data])
    labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
    labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids']
    sent_ids = [x[3] for x in all_data]

    cloze_style_sents = [make_cloze_prompt(s1, s2) for (s1, s2) in zip(sent1, sent2)]
    # Cap length to bound MPS memory and avoid allocator fragmentation on long runs.
    # Truncate from the LEFT so the cloze template ending (where the model answers
    # yes/no) is always preserved.
    self.tokenizer.truncation_side = 'left'
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True,
                              truncation=True, max_length=128)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids
    }

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] for x in all_data]

    cloze_style_sents = [make_cloze_prompt(s1, s2) for (s1, s2) in zip(sent1, sent2)]

    # Cap length to bound MPS memory and avoid allocator fragmentation on long runs.
    # Truncate from the LEFT so the cloze template ending (where the model answers
    # yes/no) is always preserved.
    self.tokenizer.truncation_side = 'left'
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True,
                              truncation=True, max_length=128)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data


STYLE_PROMPT_PREFIX = "Modern: "
# Tag deliberately has NO trailing space: GPT-2 BPE encodes the leading space as
# part of the *next* token, so a space-terminated prompt makes the model emit junk
# on the first step. The target is joined as " {s}" (leading space) instead.
STYLE_TARGET_TAG = "\nShakespearean:"


def make_style_prompt(modern):
  """Source side of the stylistic paraphrase prompt. The model is asked to
  continue this with the Shakespearean rewrite (starting with a leading space)."""
  return f"{STYLE_PROMPT_PREFIX}{modern}{STYLE_TARGET_TAG}"


def load_style_data(split, data_dir="data/shakespeare_modern"):
  """Read the sentence-aligned modern/original files produced by
  prepare_style_data.py. Returns a list of (modern, shakespearean) tuples."""
  import os
  m_path = os.path.join(data_dir, f"{split}.modern.txt")
  o_path = os.path.join(data_dir, f"{split}.original.txt")
  with open(m_path, encoding="utf-8") as f:
    modern = [ln.rstrip("\n") for ln in f if ln.strip()]
  with open(o_path, encoding="utf-8") as f:
    original = [ln.rstrip("\n") for ln in f if ln.strip()]
  assert len(modern) == len(original), \
    f"{split}: {len(modern)} modern != {len(original)} original lines"
  pairs = list(zip(modern, original))
  print(f"Loaded {len(pairs)} {split} style pairs from {data_dir}")
  return pairs


class StyleParaphraseDataset(Dataset):
  """Modern -> Shakespearean style transfer (conditional generation).

  Each example is encoded as a single sequence:
      Modern: {modern}\nShakespearean: {shakespeare}<eos>
  collate_fn returns token_ids / attention_mask plus `prompt_lens`: the token
  length of the source side ("Modern: {modern}\nShakespearean: ") for each
  example, so the training loop can mask the loss to the target side only.
  """

  def __init__(self, pairs, args=None, max_length=128):
    self.pairs = pairs
    self.p = args
    self.max_length = max_length
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    return self.pairs[idx]

  def collate_fn(self, all_data):
    eos = self.tokenizer.eos_token
    prompts = [make_style_prompt(m) for (m, _) in all_data]
    # Leading space before the target so GPT-2 tokenizes it as a normal " word".
    fulls = [make_style_prompt(m) + " " + s + eos for (m, s) in all_data]

    # prompt token length per example (no padding) -> where the target begins.
    self.tokenizer.truncation_side = 'right'
    prompt_lens = [
      len(self.tokenizer(p, truncation=True, max_length=self.max_length)['input_ids'])
      for p in prompts
    ]

    encoding = self.tokenizer(fulls, return_tensors='pt', padding=True,
                              truncation=True, max_length=self.max_length)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    return {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'prompt_lens': prompt_lens,
      'modern': [m for (m, _) in all_data],
      'shakespeare': [s for (_, s) in all_data],
    }


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data
