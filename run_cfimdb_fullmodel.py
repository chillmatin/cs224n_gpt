"""Run full-model fine-tuning on CFIMDB only and generate predictions."""
import sys
sys.path.insert(0, '.')

from types import SimpleNamespace
import random, numpy as np
import torch
from classifier import train, test, seed_everything

seed_everything(11711)

config = SimpleNamespace(
    filepath='cfimdb-classifier.pt',
    lr=1e-5,
    use_gpu=True,
    epochs=5,
    batch_size=4,       # M4 Pro 24GB; full-model + long CFIMDB reviews → keep small
    hidden_dropout_prob=0.3,
    train='data/ids-cfimdb-train.csv',
    dev='data/ids-cfimdb-dev.csv',
    test='data/ids-cfimdb-test-student.csv',
    fine_tune_mode='full-model',
    dev_out='predictions/full-model-cfimdb-dev-out.csv',
    test_out='predictions/full-model-cfimdb-test-out.csv',
)

print('Training CFIMDB full-model...')
train(config)
print('Evaluating CFIMDB full-model...')
test(config)
print('Done.')
