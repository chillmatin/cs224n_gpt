#!/usr/bin/env python3
"""
Controlled CFIMDB effective-batch-size experiment (addresses the report's
hypothesis that the CFIMDB full-model gap, 0.967 vs handout 0.976, is caused by
the small batch we needed for MPS memory).

Holds EVERYTHING fixed (same seed, micro-batch=4, lr, epochs, dropout, max_len)
and varies ONLY the effective batch size via gradient accumulation:
  * accum_steps=1  -> effective batch 4   (the small-batch baseline; also the
                       refactor sanity check -- should match the original ~0.967)
  * accum_steps=8  -> effective batch 32  (the treatment)

Same micro-batch in both, so memory footprint and per-forward numerics are
identical; only the optimizer step frequency changes. Does NOT touch the
committed cfimdb-classifier.pt or the committed predictions/.

    conda activate cs224n_dfp
    PYTORCH_ENABLE_MPS_FALLBACK=1 python run_cfimdb_batch.py
"""
import os
import sys
from types import SimpleNamespace

from classifier import train, seed_everything

SCRATCH = os.environ.get("CFIMDB_EXP_DIR", "/tmp/cfimdb_batch_exp")
os.makedirs(SCRATCH, exist_ok=True)


def run(accum_steps):
  seed_everything(11711)  # identical init/shuffle for every effective batch size
  cfg = SimpleNamespace(
      filepath=os.path.join(SCRATCH, f"cfimdb-eff{4*accum_steps}.pt"),
      lr=1e-5,
      use_gpu=True,
      epochs=5,
      batch_size=4,
      accum_steps=accum_steps,
      hidden_dropout_prob=0.3,
      fine_tune_mode="full-model",
      train="data/ids-cfimdb-train.csv",
      dev="data/ids-cfimdb-dev.csv",
      test="data/ids-cfimdb-test-student.csv",
      dev_out=os.path.join(SCRATCH, f"dev-eff{4*accum_steps}.csv"),
      test_out=os.path.join(SCRATCH, f"test-eff{4*accum_steps}.csv"),
  )
  print(f"\n########## accum_steps={accum_steps}  effective_batch={4*accum_steps} ##########")
  best = train(cfg)
  print(f"[RESULT] accum_steps={accum_steps} effective_batch={4*accum_steps} "
        f"best_dev_acc={best:.4f}")
  return best


if __name__ == "__main__":
  which = sys.argv[1] if len(sys.argv) > 1 else "all"
  results = {}
  if which in ("all", "1"):
    results[4] = run(1)
  if which in ("all", "8"):
    results[32] = run(8)
  print("\n===== CFIMDB effective-batch sweep summary =====")
  for eff in sorted(results):
    print(f"  effective batch {eff:>3}: dev acc = {results[eff]:.4f}")
