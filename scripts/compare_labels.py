#!/usr/bin/env python3
"""Compare two label files position-by-position and report agreement.

K-means cluster ids from two runs can be permuted even when the clustering is
identical, so we report BOTH:
  * raw identical fraction (how many positions have the exact same id), and
  * best-permutation agreement (match ids via a greedy majority mapping),
    which is the meaningful "same clustering?" number.
"""
import sys
from collections import Counter, defaultdict


def read(path):
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def main():
    a = read(sys.argv[1])
    b = read(sys.argv[2])
    n = len(a)
    raw = sum(x == y for x, y in zip(a, b)) / n

    # Greedy map b-ids -> a-ids by majority overlap.
    hist = defaultdict(Counter)
    for x, y in zip(b, a):
        hist[x][y] += 1
    mapping = {bi: cnt.most_common(1)[0][0] for bi, cnt in hist.items()}
    mapped = sum(mapping[y] == x for x, y in zip(a, b)) / n

    print(f"raw identical       : {raw:.4f}")
    print(f"best-permutation    : {mapped:.4f}")


if __name__ == "__main__":
    main()
