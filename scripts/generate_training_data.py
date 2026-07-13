"""Generate synthetic transfer-function training data.

Usage:
    uv run python scripts/generate_training_data.py --scenes 3000 \
        --out data/synthetic/train.h5 --workers 14 --seed 0
"""

import argparse
import os
import time
from multiprocessing import Pool

from antiphon.model.dataset import generate_scene, write_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', type=int, default=100)
    parser.add_argument('--out', type=str, default='data/synthetic/train.h5')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument('--seed', type=int, default=0,
                        help='base seed; scene i uses seed base+i')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seeds = [args.seed + i for i in range(args.scenes)]

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        scenes = []
        for i, scene in enumerate(pool.imap(generate_scene, seeds, chunksize=4)):
            scenes.append(scene)
            if (i + 1) % 50 == 0:
                el = time.perf_counter() - t0
                rate = (i + 1) / el
                eta = (args.scenes - i - 1) / rate
                print(f'{i+1}/{args.scenes} scenes '
                      f'({rate:.1f}/s, ETA {eta/60:.1f} min)', flush=True)

    # imap preserves seed order, so output is deterministic
    write_dataset(args.out, scenes)
    el = time.perf_counter() - t0
    print(f'Wrote {args.scenes} scenes '
          f'({args.scenes * scenes[0]["rcv"].shape[0]} samples) '
          f'to {args.out} in {el/60:.1f} min')


if __name__ == '__main__':
    main()
