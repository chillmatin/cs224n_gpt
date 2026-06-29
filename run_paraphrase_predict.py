"""Kayıtlı paraphrase checkpoint'inden dev/test prediction'larını üretir (eğitim YOK).

Kullanım:
    python run_paraphrase_predict.py
"""
import sys
sys.path.insert(0, '.')

from types import SimpleNamespace
from paraphrase_detection import test, add_arguments

args = SimpleNamespace(
    para_dev='data/quora-dev.csv',
    para_test='data/quora-test-student.csv',
    para_dev_out='predictions/para-dev-output.csv',
    para_test_out='predictions/para-test-output.csv',
    use_gpu=True,
    batch_size=16,
    model_size='gpt2',
    filepath='3-1e-05-paraphrase.pt',   # epoch-0 best checkpoint (dev acc ~0.863)
)
args = add_arguments(args)
test(args)
print('DONE: predictions written -> predictions/para-dev-output.csv, predictions/para-test-output.csv')
