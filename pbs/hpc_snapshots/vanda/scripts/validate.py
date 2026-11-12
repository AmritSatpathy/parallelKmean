#!/usr/bin/env python3
"""Validate K-means output against ground-truth labels.

K-means labels are arbitrary (cluster "3" has no relation to true class "3"),
so we score with permutation-invariant measures:

  * Purity: assign each predicted cluster to its majority true class, then
    accuracy of that mapping. 1.0 = perfectly recovers the classes.
  * Also prints how many predicted clusters are "pure" and a compact
    cluster -> majority-class table.

Two modes:
  1. --truth-col C  : read ground-truth from column C of the ORIGINAL csv.
  2. --truth FILE   : read ground-truth from a one-label-per-line file.

Predicted labels come from --pred FILE (one integer per line, as written by the
C++ programs via --out-labels).

Usage:
    python3 validate.py --pred out.labels --csv data/blobs.csv --truth-col 0
"""
import argparse
from collections import Counter, defaultdict


def read_lines_int(path):
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip() != ""]


def read_truth_from_csv(path, col):
    truth = []
    first = True
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.replace("\t", ",").split(",")
            if first:
                first = False
                # Skip a header row if the target column isn't an integer.
                try:
                    int(float(toks[col]))
                except ValueError:
                    continue
            truth.append(int(float(toks[col])))
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predicted labels, one per line")
    ap.add_argument("--csv", help="original csv (for --truth-col)")
    ap.add_argument("--truth-col", type=int, help="ground-truth column in --csv")
    ap.add_argument("--truth", help="ground-truth labels file, one per line")
    args = ap.parse_args()

    pred = read_lines_int(args.pred)

    if args.truth:
        truth = read_lines_int(args.truth)
    elif args.csv is not None and args.truth_col is not None:
        truth = read_truth_from_csv(args.csv, args.truth_col)
    else:
        ap.error("provide either --truth FILE or (--csv and --truth-col)")

    if len(pred) != len(truth):
        raise SystemExit(f"length mismatch: pred={len(pred)} truth={len(truth)}")

    n = len(pred)

    # Build cluster -> true-class histogram.
    hist = defaultdict(Counter)
    for p, t in zip(pred, truth):
        hist[p][t] += 1

    # Purity: sum of per-cluster majority counts / n.
    correct = 0
    mapping = {}
    for cl, cnt in hist.items():
        maj_class, maj_n = cnt.most_common(1)[0]
        mapping[cl] = (maj_class, maj_n, sum(cnt.values()))
        correct += maj_n
    purity = correct / n

    n_clusters = len(hist)
    n_classes = len(set(truth))
    pure = sum(1 for cl in hist if len(hist[cl]) == 1)

    print(f"points          : {n}")
    print(f"predicted clust : {n_clusters}")
    print(f"true classes    : {n_classes}")
    print(f"purity          : {purity:.4f}")
    print(f"fully-pure clust: {pure}/{n_clusters}")
    print()
    print(f"{'cluster':>7} {'maj_class':>9} {'maj_frac':>9} {'size':>8}")
    for cl in sorted(mapping):
        mc, mn, sz = mapping[cl]
        print(f"{cl:>7} {mc:>9} {mn / sz:>9.3f} {sz:>8}")


if __name__ == "__main__":
    main()
