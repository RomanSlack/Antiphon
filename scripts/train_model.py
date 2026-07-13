"""Train foundation model v1.

Usage:
    uv run python scripts/train_model.py --data data/synthetic/train.h5 \
        --out data/runs/v1 --epochs 30
"""

import argparse

from antiphon.model.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/synthetic/train.h5')
    parser.add_argument('--out', type=str, default='data/runs/v1')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-scenes', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    train(args.data, args.out, epochs=args.epochs,
          batch_scenes=args.batch_scenes, lr=args.lr,
          seed=args.seed, threads=args.threads)


if __name__ == '__main__':
    main()
