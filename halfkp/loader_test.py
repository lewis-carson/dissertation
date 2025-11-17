#!/usr/bin/env python3
import time
from pathlib import Path

from train import create_sparse_dataloader, DataloaderSkipConfig


def run_test(batch_size=1024, concurrency=4, max_batches=200, cyclic=False):
    data_dir = Path(__file__).resolve().parent / "data"
    binpack_paths = [str(p) for p in data_dir.glob("*.binpack")]
    print(f"Using files: {binpack_paths}")
    skip_config = DataloaderSkipConfig()

    dl = create_sparse_dataloader(
        binpack_paths,
        batch_size=batch_size,
        skip_config=skip_config,
        num_workers=concurrency,
        cyclic=cyclic,
    )

    start = time.time()
    i = 0
    for batch in dl:
        i += 1
        if i % 10 == 0:
            print(f"Batch {i} at {(time.time() - start):.3f}s")
        if i >= max_batches:
            break
    print(f"Done - iterated {i} batches in {time.time() - start:.3f} seconds")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--max-batches', type=int, default=200)
    p.add_argument('--cyclic', action='store_true')
    args = p.parse_args()

    run_test(batch_size=args.batch_size, concurrency=args.concurrency, max_batches=args.max_batches, cyclic=args.cyclic)
