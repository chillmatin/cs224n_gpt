"""Run paraphrase detection training and generate predictions.

SMOKE=True  -> küçük altküme + 1 epoch ile hızlı pipeline doğrulaması (predictions ÜRETMEZ).
SMOKE=False -> tam eğitim (3 epoch, tüm veri) + dev/test prediction dosyaları.
"""
import sys
sys.path.insert(0, '.')

from paraphrase_detection import train, test, seed_everything, add_arguments
from types import SimpleNamespace

SMOKE = False   # <<< tam koşu için False yap

seed_everything(11711)

args = SimpleNamespace(
    para_train='data/quora-train.csv',
    para_dev='data/quora-dev.csv',
    para_test='data/quora-test-student.csv',
    para_dev_out='predictions/para-dev-output.csv',
    para_test_out='predictions/para-test-output.csv',
    seed=11711,
    epochs=1 if SMOKE else 3,
    use_gpu=True,
    batch_size=16,            # M4 Pro 24GB; OOM olursa 8'e düşür
    lr=1e-5,
    model_size='gpt2',
    smoke_limit=2000 if SMOKE else None,
)
args = add_arguments(args)
args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'

print(f'Training paraphrase detection... (SMOKE={SMOKE})')
train(args)

if SMOKE:
    print('[SMOKE] Eğitim pipeline OK. Tam koşu + predictions için SMOKE=False yap.')
else:
    print('Evaluating paraphrase detection...')
    test(args)
    print('Done.')
