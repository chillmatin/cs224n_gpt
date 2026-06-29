#!/usr/bin/env python3
"""
Measure our sonnet model's chrF on the DEV held-out set, using the gold file the
course provides (data/TRUE_sonnets_held_out_dev.txt). Generation reuses the
already-trained checkpoint (9_10-1e-05-sonnet.pt) -- NO retraining.

  python eval_sonnet_dev.py --use_gpu
  python eval_sonnet_dev.py --use_gpu --temperature 1.0 --top_p 0.9   # tuning
"""

import argparse
import os
from sonnet_generation import generate_submission_sonnets, seed_everything
from evaluation import test_sonnet


def get_args():
  p = argparse.ArgumentParser()
  p.add_argument("--use_gpu", action='store_true')
  p.add_argument("--epochs", type=int, default=10)          # -> checkpoint 9_10-1e-05-sonnet.pt
  p.add_argument("--lr", type=float, default=1e-5)
  p.add_argument("--temperature", type=float, default=1.2)  # same as the test run
  p.add_argument("--top_p", type=float, default=0.9)
  p.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  p.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dev.txt")
  p.add_argument("--gold_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  p.add_argument("--seed", type=int, default=11711)
  return p.parse_args()


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # generate_submission_sonnets loads {epochs-1}_{filepath}
  seed_everything(args.seed)
  generate_submission_sonnets(args)
  # Only score when the held-out seed actually matches the gold (the dev set).
  # When regenerating the TEST submission file there is no matching gold, so skip
  # scoring instead of producing a meaningless dev-gold-vs-test-gen number.
  if 'dev' not in args.held_out_sonnet_path:
    args.gold_path = ''
  if os.path.exists(args.gold_path):
    score = test_sonnet(test_path=args.sonnet_out, gold_path=args.gold_path)
    print(f"\n=== Sonnet chrF (temp={args.temperature}, top_p={args.top_p}) ===")
    print(f"  chrF = {score:.2f}")
  else:
    print(f"\n[no gold at {args.gold_path}] generated only -> {args.sonnet_out} "
          f"(temp={args.temperature}, top_p={args.top_p})")
