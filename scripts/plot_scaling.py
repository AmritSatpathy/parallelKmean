#!/usr/bin/env python3
"""Derive speedup/efficiency from a scaling CSV and (optionally) plot figures.

Input CSV columns (as produced by run_scaling.sh):
    impl,nproc,n,dims,k,iters,inertia,time

Always writes results/scaling_metrics.csv with derived columns:
    impl,nproc,time,speedup,efficiency
(speedup is relative to the serial baseline time; if no 'serial' row exists,
the single-process MPI time is used as the baseline.)

If matplotlib is available, also writes:
    results/speedup.png, results/efficiency.png, results/runtime.png

Standard-library only for the metrics; matplotlib only for the pictures.

Usage:
    python3 plot_scaling.py results/scaling_mpi.csv
"""
import csv
import os
import sys


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["nproc"] = int(r["nproc"])
            r["time"] = float(r["time"])
            rows.append(r)
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/scaling_mpi.csv"
    rows = load(path)
    os.makedirs("results", exist_ok=True)

    # Baseline: serial if present, else the smallest-nproc parallel row.
    serial = [r for r in rows if r["impl"] == "serial"]
    if serial:
        base_t = serial[0]["time"]
    else:
        par = sorted((r for r in rows if r["impl"] != "serial"),
                     key=lambda r: r["nproc"])
        base_t = par[0]["time"]

    par = sorted((r for r in rows if r["impl"] != "serial"),
                 key=lambda r: r["nproc"])

    out = "results/scaling_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "nproc", "time", "speedup", "efficiency"])
        print(f"{'impl':>8} {'np':>3} {'time':>9} {'speedup':>8} {'eff':>6}")
        for r in par:
            sp = base_t / r["time"]
            eff = sp / r["nproc"]
            w.writerow([r["impl"], r["nproc"], f"{r['time']:.6f}",
                        f"{sp:.4f}", f"{eff:.4f}"])
            print(f"{r['impl']:>8} {r['nproc']:>3} {r['time']:>9.4f} "
                  f"{sp:>8.3f} {eff:>6.3f}")
    print(f"\nwrote {out}  (baseline time = {base_t:.4f}s)")

    # Optional figures.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: wrote metrics CSV only "
              "(plot it in Excel or install matplotlib).")
        return

    ps = [r["nproc"] for r in par]
    ts = [r["time"] for r in par]
    sp = [base_t / r["time"] for r in par]
    eff = [s / p for s, p in zip(sp, ps)]

    plt.figure()
    plt.plot(ps, sp, "o-", label="measured")
    plt.plot(ps, ps, "k--", label="ideal (linear)")
    plt.xlabel("MPI processes")
    plt.ylabel("speedup")
    plt.title("Strong-scaling speedup")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/speedup.png", dpi=130, bbox_inches="tight")

    plt.figure()
    plt.plot(ps, eff, "o-")
    plt.axhline(1.0, color="k", ls="--")
    plt.ylim(0, 1.1)
    plt.xlabel("MPI processes")
    plt.ylabel("efficiency")
    plt.title("Strong-scaling efficiency")
    plt.grid(True)
    plt.savefig("results/efficiency.png", dpi=130, bbox_inches="tight")

    plt.figure()
    plt.loglog(ps, ts, "o-")
    plt.xlabel("MPI processes")
    plt.ylabel("cluster time (s)")
    plt.title("Runtime vs processes")
    plt.grid(True, which="both")
    plt.savefig("results/runtime.png", dpi=130, bbox_inches="tight")

    print("wrote results/speedup.png, efficiency.png, runtime.png")


if __name__ == "__main__":
    main()
