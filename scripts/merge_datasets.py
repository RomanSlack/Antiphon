"""Concatenate scene HDF5 datasets produced by generate_training_data.py.

Usage:
    uv run python scripts/merge_datasets.py out.h5 in1.h5 in2.h5 [...]
"""

import sys

import h5py
import numpy as np


def main():
    out_path, in_paths = sys.argv[1], sys.argv[2:]
    parts = {}
    attrs = None
    for p in in_paths:
        with h5py.File(p, 'r') as f:
            if attrs is None:
                attrs = {k: f.attrs[k] for k in f.attrs}
            else:
                assert np.allclose(attrs['freqs'], f.attrs['freqs'])
            for k in f.keys():
                parts.setdefault(k, []).append(f[k][:])

    seeds = np.concatenate(parts['seed'])
    assert len(np.unique(seeds)) == len(seeds), 'duplicate scene seeds'

    with h5py.File(out_path, 'w') as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        for k, arrs in parts.items():
            f.create_dataset(k, data=np.concatenate(arrs))
    print(f'wrote {out_path}: {len(seeds)} scenes')


if __name__ == '__main__':
    main()
