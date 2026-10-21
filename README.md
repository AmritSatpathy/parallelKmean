# Parallel K-means Clustering (Serial / MPI / MPI+CUDA)

An implementation of Lloyd's K-means clustering algorithm in C++, written three
ways so their parallel scaling and efficiency can be compared:

| Build target | Source | Parallelism |
|--------------|--------|-------------|
| `serial` | `src/kmeans_serial.cpp` | none — reference baseline |
| `mpi`    | `src/kmeans_mpi.cpp` | MPI, data-parallel over points (CPU) |
| `cuda`   | `src/kmeans_mpi_cuda.cu` | MPI across ranks + CUDA within each rank (GPU) |

All three share `src/common.hpp` and use the **same k-means++ seeding**, so for a
given dataset, `K`, and `--seed` they converge to the same clustering. That is
how the parallel versions are validated: their output must match the serial
baseline.

**Why this is a non-trivial parallelization.** The assignment step (each point to
its nearest centroid) is embarrassingly parallel. The centroid-update step is a
**global reduction** over all points — in MPI it is one `MPI_Allreduce` per
iteration; on the GPU it is `atomicAdd` accumulation plus the same cross-rank
`MPI_Allreduce`. That reduction and its synchronization is the interesting part
of the scaling study.

## Repository layout

```
src/        C++ sources (serial, MPI, CUDA) + shared header
scripts/    data generator + download, validation, scaling harness, plotting
pbs/        PBS (qsub) batch scripts for Atlas (CPU) and Vanda (GPU)
report/     the project report (Markdown; export to PDF for submission)
Makefile    build targets
.gitignore  excludes build artifacts, datasets, and results
```

`bin/`, `data/`, and `results/` are created on demand and are git-ignored (they
hold build outputs, large datasets, and generated figures — all reproducible
from the commands below).

### What each file does

**Source (`src/`)**

- `common.hpp` — shared utilities: CSV reader (with optional label column),
  squared-distance, deterministic k-means++ seeding, result writers, timing.
  Points are stored row-major in a flat `float` array, the layout the MPI
  scatter and CUDA kernels expect.
- `kmeans_serial.cpp` — single-threaded Lloyd's algorithm. The correctness
  reference and the denominator for speedup.
- `kmeans_mpi.cpp` — distributes points across ranks with `MPI_Scatterv`; each
  rank assigns its local points and forms partial sums; `MPI_Allreduce` combines
  them so every rank recomputes identical centroids.
- `kmeans_mpi_cuda.cu` — same MPI decomposition between ranks, but each rank runs
  the assignment step on a GPU (one CUDA thread per point, centroids staged in
  shared memory, partial sums via `atomicAdd`), then the same cross-rank
  `MPI_Allreduce`.

**Scripts (`scripts/`)** — Python scripts use only the standard library (except
plotting, which needs `matplotlib`); shell scripts are POSIX `bash`.

- `gen_blobs.cpp` — fast C++ generator for synthetic Gaussian-blob datasets with
  known ground-truth labels; produces multi-million-point files in seconds. It
  is compiled via `make gen` to `bin/gen_blobs`.
- `fetch_mnist.sh` — downloads MNIST in CSV form to `data/mnist.csv`
  (70,000 rows: label in column 0, then 784 pixel features).
- `validate.py` — scores predicted clusters against ground-truth labels using
  purity and a per-cluster majority-class table.
- `compare_labels.py` — compares two label files position-by-position (raw and
  best-permutation agreement); used to confirm parallel output matches serial.
- `run_scaling.sh` — runs the serial baseline and MPI at a list of process
  counts, writing `results/scaling_mpi.csv`.
- `plot_scaling.py` — turns a scaling CSV into `results/scaling_metrics.csv`
  (speedup, efficiency) and, if `matplotlib` is installed, PNG figures.

**PBS job scripts (`pbs/`)** — Atlas and Vanda use PBS (`qsub`), not Slurm.

- `atlas_mpi.pbs` — one MPI (CPU) strong-scaling run on Atlas (CPU-only
  cluster); submit once per process count.
- `vanda_cuda.pbs` — an MPI+CUDA run on Vanda (CPU+GPU cluster), one rank per
  GPU.

Both auto-generate the benchmark dataset if it is missing. Edit the
`module load` lines (and `CUDA_ARCH` for Vanda) to match each cluster.

## Prerequisites

- A C++17 compiler (`g++`), an MPI implementation (`mpicxx` + `mpirun`; tested
  with Open MPI 4.1), and for the GPU build the CUDA toolkit (`nvcc`) with a
  driver new enough for that toolkit version.
- Python 3 for the validation and scaling scripts. `matplotlib` is optional and
  only needed to render figures (the metrics CSV is always written without it).

## Build

```bash
make serial                 # -> bin/kmeans_serial
make mpi                    # -> bin/kmeans_mpi
make cuda CUDA_ARCH=sm_86   # -> bin/kmeans_mpi_cuda  (set arch for your GPU)
make gen                    # -> bin/gen_blobs
make all                    # all of the above
```

Pick `CUDA_ARCH` for the target GPU: `sm_70` (V100), `sm_80` (A100),
`sm_86` (RTX 30xx), `sm_89` (RTX 40xx), `sm_90` (H100).

## Get data

**Real data — MNIST** (label in column 0):

```bash
bash scripts/fetch_mnist.sh          # -> data/mnist.csv
```

**Synthetic data** with known ground truth (label in column 0):

```bash
# Large benchmark set: 3M points x 64 dims, K=128 (~1.7 GB; serial run ~10 min)
./bin/gen_blobs --n 3000000 --dims 64 --k 128 --sep 25 --out data/blobs_big.csv
```

Any numeric CSV works. Pass `--label-col C` to the K-means programs to exclude
column `C` as a ground-truth label (omit the flag if there is no label column).

The serial cost per iteration is `n · K · dims`; the dataset above is sized so
the serial baseline runs ~10 minutes, keeping timing dominated by real compute
rather than start-up overhead. Scale `n` to hit a different target.

## Run the project

```bash
# Serial baseline
./bin/kmeans_serial data/mnist.csv 10 --label-col 0 --seed 42 \
    --out-labels results/serial.labels

# MPI (CPU), 8 ranks
mpirun -np 8 ./bin/kmeans_mpi data/mnist.csv 10 --label-col 0 --seed 42 \
    --out-labels results/mpi.labels

# MPI + CUDA, one rank per GPU
mpirun -np 1 ./bin/kmeans_mpi_cuda data/mnist.csv 10 --label-col 0 --seed 42 \
    --out-labels results/cuda.labels
```

Options accepted by all three programs:

| Option | Meaning |
|--------|---------|
| `<csv> <k>` | input file and number of clusters (positional, required) |
| `--label-col C` | treat column `C` as a ground-truth label, excluded from features |
| `--max-iter N` | maximum iterations (default 100) |
| `--tol T` | stop when the relative inertia change drops below `T` |
| `--fixed-iter` | ignore `--tol` and always run exactly `--max-iter` iterations |
| `--seed S` | k-means++ seed (default 1234) |
| `--out-labels F` | write the per-point cluster assignment to `F` |
| `--out-centroids F` | write the final centroids (CSV) to `F` |

Each run prints a `RESULT` line (`n dims k nproc iters inertia time`) that the
scaling harness parses.

Use `--fixed-iter` for benchmarking: every process count then performs identical
work, so speedup and efficiency reflect the parallelization rather than how many
iterations a particular seed happened to need.

## Validate correctness

```bash
# Clustering quality vs ground-truth labels (purity, per-cluster majority class)
python3 scripts/validate.py --pred results/mpi.labels \
    --csv data/mnist.csv --truth-col 0

# Parallel output vs serial baseline (expect agreement ~1.0)
python3 scripts/compare_labels.py results/serial.labels results/mpi.labels
```

## Scaling study

```bash
# Serial baseline + MPI at each process count, fixed 32 iterations.
FIXEDITER=1 MAXIT=32 REPS=1 \
    bash scripts/run_scaling.sh data/blobs_big.csv 128 "1 2 4 8 16"
#   -> results/scaling_mpi.csv

# Derive speedup/efficiency (always) and figures (if matplotlib is installed).
python3 scripts/plot_scaling.py results/scaling_mpi.csv
#   -> results/scaling_metrics.csv, results/{speedup,efficiency,runtime}.png
```

Environment variables the harness understands: `FIXEDITER` (0/1), `MAXIT`,
`REPS` (repeats; the best time is kept), `SEED`, `LABELCOL`.

To render figures without altering a system Python installation, create a local
virtual environment first:

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install matplotlib
python scripts/plot_scaling.py results/scaling_mpi.csv
```

## Running on Atlas / Vanda (PBS / qsub)

Atlas and Vanda use PBS. Atlas is CPU-only (run the MPI scaling there); Vanda
has GPUs (run the MPI+CUDA study there). Edit the `module load` lines in the
`pbs/` scripts, and `CUDA_ARCH` in the Vanda script, to match each cluster.

**Copy the project to the cluster** (binaries and data are rebuilt/regenerated
there, so they are excluded):

```bash
rsync -av --exclude 'bin/' --exclude 'data/' --exclude 'results/' \
    --exclude '.venv/' --exclude '*.labels' --exclude 'report/' \
    --exclude 'pbs/logs/' \
    ./ satpathy.amrit@atlas8.nus.edu.sg:~/kMeans/
```

**Build.** On Atlas, compile on the login node:

```bash
source /app1/ebenv
module load <gcc> <openmpi>          # names from `module avail`
make mpi gen
```

On Vanda the login node is a VM, so compile on a compute node:

```bash
qsub -I -l select=1:ncpus=1:ngpus=1 -l walltime=00:10:00   # interactive node
module load <gcc> <openmpi> <cuda>
make cuda gen CUDA_ARCH=sm_XX        # set arch to Vanda's GPU
exit
```

**Submit the scaling jobs:**

```bash
# Atlas — MPI (CPU) strong scaling, one job per process count
for p in 1 2 4 8 16 32 64; do
    qsub -l select=1:ncpus=$p:mpiprocs=$p -v NP=$p pbs/atlas_mpi.pbs
done

# Vanda — MPI + CUDA, one rank per GPU (1, 2, 4 GPUs)
qsub -l select=1:ncpus=1:ngpus=1:mpiprocs=1 -v NP=1 pbs/vanda_cuda.pbs
qsub -l select=1:ncpus=2:ngpus=2:mpiprocs=2 -v NP=2 pbs/vanda_cuda.pbs
qsub -l select=2:ncpus=2:ngpus=2:mpiprocs=2 -v NP=4 pbs/vanda_cuda.pbs
```

Logs are written under `pbs/logs/`. Collect the `RESULT` lines from the logs
into a CSV with columns `impl,nproc,n,dims,k,iters,inertia,time`, then reuse
`plot_scaling.py` to make the figures.

## End-to-end example (quick validation)

```bash
make all
bash scripts/fetch_mnist.sh
./bin/kmeans_serial data/mnist.csv 10 --label-col 0 --seed 42 \
    --out-labels results/serial.labels
mpirun -np 8 ./bin/kmeans_mpi data/mnist.csv 10 --label-col 0 --seed 42 \
    --out-labels results/mpi.labels
python3 scripts/compare_labels.py results/serial.labels results/mpi.labels
python3 scripts/validate.py --pred results/mpi.labels \
    --csv data/mnist.csv --truth-col 0
```

