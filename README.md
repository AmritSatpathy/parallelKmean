# Parallel K-means with MPI and CUDA

DSA5208 · Scalable Distributed Computing for Data Science · NUS · November 2026

| Team member | Student ID |
| --- | --- |
| Amrit Satpathy | A0355181E |
| Ritik Srivastava | A0355860X |

Repository: <https://github.com/AmritSatpathy/parallelKmean>
Canonical branch: [`mainline`](https://github.com/AmritSatpathy/parallelKmean/tree/mainline). The submission ZIP is a frozen source snapshot with the run documentation and checksum tools; dataset CSVs are downloaded or generated separately.

Start with [the final PDF report](report/Parallel_Kmeans_Project_Report.pdf). It explains Lloyd's algorithm, MPI decomposition and collectives, CUDA kernels, validation, scaling, MNIST interpretation, limitations, references, and AI usage. All results are one team study. Only completed experiments are used in the report.

## Table of contents

1. [Start here: choose A or B](#1-start-here-choose-a-or-b)
2. [Common setup after A or B](#2-common-setup-after-a-or-b)
3. [Files and evidence](#3-files-and-evidence)
4. [Local CPU validation](#4-local-cpu-validation)
5. [Datasets and full workload](#5-datasets-and-full-workload)
6. [Copy to Atlas or Vanda](#6-copy-to-atlas-or-vanda)
7. [Submit on Atlas](#7-submit-on-atlas)
8. [Submit on Vanda](#8-submit-on-vanda)
9. [Monitor and collect results](#9-monitor-and-collect-results)
10. [Recorded results](#10-recorded-results)
11. [Optional analysis and report tools](#11-optional-analysis-and-report-tools)

## 1. Start here: choose A or B

| Starting point | When to use it | First command |
| --- | --- | --- |
| **A — submitted ZIP** | You have the code ZIP from Canvas and want the exact submitted files. | `unzip DSA5208_Parallel_Kmeans_A0355181E_A0355860X_Code.zip` |
| **B — public GitHub repository** | You want the current `mainline` source from GitHub. | `git clone --branch mainline https://github.com/AmritSatpathy/parallelKmean.git` |

Choose **one** path. Run its commands in a fresh directory, then continue at [Section 2](#2-common-setup-after-a-or-b). The ZIP and the public repository both place the README at `parallelKmean/README.md`, next to the Makefile.

### 1A. Extract the submitted ZIP and enter the project

From a terminal in the folder containing the downloaded ZIP:

```bash
unzip DSA5208_Parallel_Kmeans_A0355181E_A0355860X_Code.zip
cd parallelKmean
```

### 1B. Clone `mainline` and enter the project

From a terminal in the folder where you want the project:

```bash
git clone --branch mainline https://github.com/AmritSatpathy/parallelKmean.git
cd parallelKmean
```

## 2. Common setup after A or B

All commands below run from the `parallelKmean` directory. Both paths converge here. Requirements: Bash, Python 3, curl or wget, GNU Make, a C++17 compiler, and Open MPI (`mpicxx`, `mpirun`). GPU runs additionally need CUDA and an NVIDIA GPU. Dataset CSVs are excluded from the ZIP and Git repository.

### 2.1 Verify the source snapshot (ZIP path only)

Checks every packaged file against `SUBMISSION_MANIFEST.json`. Run this after **1A**; a Git clone has no submission manifest.

```bash
python3 scripts/verify_submission.py
```

### 2.2 Download the public MNIST files

Downloads train and test CSVs into `data/`, verifies each checksum, and creates the exact combined `data/mnist.csv` used for the project.

```bash
bash scripts/fetch_mnist.sh
```

The verified sources are the [MNIST-in-CSV page](https://pjreddie.com/projects/mnist-in-csv/), [training CSV](https://data.pjreddie.com/files/mnist_train.csv) and [test CSV](https://data.pjreddie.com/files/mnist_test.csv). The script tries a [GitHub mirror](https://github.com/halimb/MNIST-csv) if needed and accepts it only when the checksums match. The public files were checked byte-for-byte against the project inputs on 12 November 2026.

### 2.3 Check the downloaded dataset

Verifies all three MNIST CSVs against the project's recorded sizes and SHA-256 checksums.

```bash
python3 scripts/verify_data.py
```

### 2.4 Build the CPU programs

Compiles the serial implementation, MPI implementation, and synthetic-data generator. Run on the machine where the binaries will execute; on HPC, the PBS scripts build inside the compute-node allocation.

```bash
make serial mpi gen
```

### 2.5 Run the serial MNIST example

Creates an output directory and runs ten fixed iterations on the downloaded 70,000-row dataset. Column 0 is the digit label and is excluded from features.

```bash
mkdir -p results/reproduction
./bin/kmeans_serial data/mnist.csv 10 --label-col 0 --seed 42 --max-iter 10 --fixed-iter
```

### 2.6 Run the two-rank MPI example

Runs the same dataset and iteration budget with two MPI processes. Compare the printed `RESULT` lines and inertia; small floating-point differences can occur.

```bash
mpirun -np 2 ./bin/kmeans_mpi data/mnist.csv 10 --label-col 0 --seed 42 --max-iter 10 --fixed-iter
```

For HPC, complete steps 2.2–2.3 locally, then use [Section 6](#6-copy-to-atlas-or-vanda) to transfer code and data. The Atlas and Vanda PBS scripts perform step 2.4 inside the job. Local validation can continue with Section 4.

## 3. Files and evidence

| Path | Purpose |
| --- | --- |
| `src/kmeans_serial.cpp` | Serial reference |
| `src/kmeans_mpi.cpp` | Distributed CPU implementation |
| `src/kmeans_mpi_cuda.cu` | MPI with CUDA assignment and accumulation |
| `src/common.hpp` | CSV loading, initialization, output helpers |
| `scripts/` | Data generator, scaling harness, validation and plotting |
| `pbs/atlas_sweep.pbs`, `pbs/vanda_cuda.pbs` | Site-specific PBS jobs; pbs/README.md documents comparison with downloaded HPC copies |
| `data/` | Data notes and expected checksums; dataset CSVs are excluded |
| `results/` | Archived benchmark CSVs; preserve these when reproducing |
| `pbs/logs/atlas/`, `pbs/logs/vanda/` | Completed job logs |
| `report/experiment_manifest.json` | Report input hashes and experiment provenance |
| `report/BUILD_REPORT.md` | Optional report regeneration instructions |

The ZIP and Git repository exclude dataset CSVs. Follow Section 2 to download and verify MNIST, or Section 5 to generate synthetic data. See [data notes](data/README.md) for verified sources and checksums. No precompiled binaries are supplied: compile for the execution machine.

### 3.1 Compiler and runtime requirements

CPU: a C++17 compiler, GNU Make, Bash and Open MPI (`mpicxx`, `mpirun`). The Atlas runs used GCC 7.3.0 and Open MPI 4.1.1. CUDA additionally requires an NVIDIA GPU, a compatible driver and CUDA toolkit; the Vanda configuration used GCC 13.2.0, Open MPI 4.1.6 and CUDA 12.4.0 with A40 GPUs (`sm_86`). Python 3 is needed for validation; plotting requires matplotlib. CUDA is optional for the CPU implementation. `make all` requires CUDA; use the explicit CPU targets below on CPU machines.

## 4. Local CPU validation

### 4.1 Generate a small labelled dataset and compare serial/MPI labels

The following commands generate 10,000 labelled points, run both implementations with the same seed, then check label agreement and cluster purity.

Run these commands from either Path A or Path B, inside `parallelKmean`. Use a fresh extraction for experiments so archived results remain intact. On HPC, run computations inside a scheduler allocation, not on the login node.

```bash
# Build programs and create an isolated output directory.
make serial mpi gen
mkdir -p data results/reproduction
# Generate labelled test data and save serial/MPI assignments.
./bin/gen_blobs --n 10000 --dims 16 --k 8 --seed 0 --sep 25 --out data/smoke.csv
./bin/kmeans_serial data/smoke.csv 8 --label-col 0 --seed 42 --max-iter 20 --fixed-iter --out-labels results/reproduction/serial.labels
mpirun -np 2 ./bin/kmeans_mpi data/smoke.csv 8 --label-col 0 --seed 42 --max-iter 20 --fixed-iter --out-labels results/reproduction/mpi.labels
# Compare assignments, then compare clusters with ground-truth labels.
python3 scripts/compare_labels.py results/reproduction/serial.labels results/reproduction/mpi.labels
python3 scripts/validate.py --pred results/reproduction/mpi.labels --csv data/smoke.csv --truth-col 0
```

`compare_labels.py` reports raw and majority-mapped agreement; the latter is a heuristic, not an optimal one-to-one label permutation. Compare equal-length label files. Floating-point reduction order can affect labels near boundaries, so matching displayed inertia alone is not proof of identical clustering.

The default compiler flags contain `-march=native`. Build on the execution machine, or use `make serial mpi gen CXXFLAGS='-O3 -std=c++17 -Wall -Wextra'` for a portable CPU build.

## 5. Datasets and full workload

MNIST has 70,000 rows, with the ground-truth digit in column 0 and 784 pixel features. Always pass `--label-col 0` so the digit is excluded from clustering features.

### 5.1 Run a longer MNIST validation

Runs the serial program until convergence or 100 iterations, writes cluster labels, and reports purity against the true digit labels. Use a suitable allocation for the longer run:

```bash
test -s data/mnist.csv
./bin/kmeans_serial data/mnist.csv 10 --label-col 0 --seed 42 --max-iter 100 --out-labels results/reproduction/mnist.labels
python3 scripts/validate.py --pred results/reproduction/mnist.labels --csv data/mnist.csv --truth-col 0
```

Omitting `--fixed-iter` permits early stopping. The synthetic CSV is excluded from the ZIP.

### 5.2 Generate the full synthetic workload

After `make gen`, generate 3,000,000 labelled points with 64 features and 128 clusters on a compute node or suitable local machine:

```bash
./bin/gen_blobs --n 3000000 --dims 64 --k 128 --spread 1 --sep 25 --seed 0 --out data/blobs_big.csv
```

Generation seed 0 and clustering seed 42 are different parameters. On 12 November, the downloaded Atlas and Vanda `blobs_big.csv` files were checked: each is 1,757,795,622 bytes, 3,000,000 rows and 65 columns, with identical SHA-256 `4c420e3b2e6b442e807a658dd8d0c3d9b1cfb685e51c8868863acc51e8d3de2b`. The large CSV is deliberately omitted from the ZIP. This verifies the downloaded datasets; the older job logs did not capture contemporaneous input hashes. Check a generated file with `python3 scripts/verify_data.py --files blobs_big.csv`. If its checksum differs, it is a new workload instance: keep its results separate from the archived measurements. The generator uses C++ random distributions, whose output may differ between standard-library implementations.

## 6. Copy to Atlas or Vanda

### 6.1 Transfer code and data to a fresh HPC directory

Run this block on your **local computer**, from the extracted `parallelKmean` folder. Connect to the NUS VPN first. Set the account and choose Atlas or Vanda. The final slash on `./` means copy the folder contents.

```bash
# Set the account, destination and new remote working directory.
HPC_USER=YOUR_NUS_ID
HPC_HOST=atlas9.nus.edu.sg
# For Vanda instead: HPC_HOST=vanda.nus.edu.sg
REMOTE_DIR=parallelKmean_reproduction
ssh "$HPC_USER@$HPC_HOST" "mkdir -p ~/$REMOTE_DIR"
# Copy project code and data without overwriting saved benchmark outputs.
rsync -av --progress \
  --exclude '.git/' --exclude 'bin/' --exclude '.venv/' \
  --exclude '__pycache__/' --exclude '.DS_Store' \
  --exclude 'results/' --exclude 'pbs/logs/' \
  ./ "$HPC_USER@$HPC_HOST:~/$REMOTE_DIR/"
```

This copies source, scripts, documentation and **all files in `data/`**, while leaving remote benchmark outputs alone. Repeat the same `rsync` command for incremental sync, only when no job is using that working directory. Do not sync code over a running job; choose a new `REMOTE_DIR` for a new experiment. There is no `--delete`, so extra remote files are preserved.

### 6.2 Sync and verify the dataset

Copy the dataset, then compare its local and remote SHA-256 hashes:

```bash
rsync -av --progress data/ "$HPC_USER@$HPC_HOST:~/$REMOTE_DIR/data/"
shasum -a 256 data/mnist.csv
ssh "$HPC_USER@$HPC_HOST" "sha256sum ~/$REMOTE_DIR/data/mnist.csv"
```

The two hashes should match.

### 6.3 Copy an existing synthetic CSV if available

Transfer the synthetic CSV if it exists on your local machine:

```bash
# Only if this file is available locally:
test -s data/blobs_big.csv && rsync -av --progress data/blobs_big.csv "$HPC_USER@$HPC_HOST:~/$REMOTE_DIR/data/"
# Compare local shasum and remote sha256sum for blobs_big.csv as above.
```

### 6.4 Reuse a synthetic CSV already on HPC

If the large synthetic dataset is already on the **same HPC system** in `~/parallelKmean/data/`, copy it into the fresh working directory there:

```bash
# Run after SSH, on the HPC system:
mkdir -p ~/parallelKmean_reproduction/data
cp -p ~/parallelKmean/data/blobs_big.csv ~/parallelKmean_reproduction/data/blobs_big.csv
sha256sum ~/parallelKmean/data/blobs_big.csv ~/parallelKmean_reproduction/data/blobs_big.csv
```

This preserves the original HPC dataset. If it is unavailable, use the documented generator and label the resulting measurements as a new reproduction.

### 6.5 Download finished job results

After completion, download results from your **local** terminal. Use a new destination each time to preserve previous runs:

```bash
DOWNLOAD_DIR="hpc_download_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DOWNLOAD_DIR/results" "$DOWNLOAD_DIR/logs"
rsync -av "$HPC_USER@$HPC_HOST:~/$REMOTE_DIR/results/" "$DOWNLOAD_DIR/results/"
rsync -av "$HPC_USER@$HPC_HOST:~/$REMOTE_DIR/pbs/logs/" "$DOWNLOAD_DIR/logs/"
```

## 7. Submit on Atlas

Connect to the NUS VPN first. First complete the copy/sync section with `HPC_HOST=atlas9.nus.edu.sg`. A local `/Users/...` path does not exist on Atlas.

### 7.1 Submit a 16-rank CPU job

SSH from your local terminal and submit from the remote copy:

```bash
# Log in and enter the separate reproduction directory.
ssh YOUR_NUS_ID@atlas9.nus.edu.sg
cd ~/parallelKmean_reproduction
mkdir -p pbs/logs results data
bash -n pbs/atlas_sweep.pbs
# Request 16 CPU slots and submit the serial-plus-MPI sweep.
JOB_ID=$(qsub -q parallel -l select=1:ncpus=16:mpiprocs=16:mem=40gb -l walltime=02:00:00 -v 'CSV=data/blobs_big.csv,K=128,PROCS=16,MAXIT=100,REPS=1,SEED=42' pbs/atlas_sweep.pbs)
printf '%s\n' "$JOB_ID"
```

The job builds on the compute node, generates `data/blobs_big.csv` if absent, and runs a serial baseline followed by 16 MPI ranks. For 12, 24, 32 or 48 ranks, change **both** `ncpus`/`mpiprocs` and `PROCS`; use a fresh working copy for each job. The recorded parallel queue minimum was 12 CPUs. Queue policies and available nodes are site-specific. Memory belongs inside the `select` request on this system. The packaged script defaults to a 16-rank job, matching this example.

For MNIST, check the file exists and change the `-v` argument to `CSV=data/mnist.csv,K=10,PROCS=16,MAXIT=32,REPS=1,SEED=42`. Missing real-data inputs must not be replaced with synthetic data.

The Atlas script pins the compiler and MPI paths used in the successful runs. If these paths change, edit the environment block to the matching installed toolchain. Do not mix MVAPICH and Open MPI libraries. Each sweep overwrites `results/scaling_mpi.csv` and `results/scaling_atlas.csv` in its working directory. A full 100-iteration serial baseline takes around 28 minutes in the recorded runs; job wall time includes loading, initialization, compilation and all runs, not just the reported MPI iteration time.

## 8. Submit on Vanda

Complete the copy/sync section with `HPC_HOST=vanda.nus.edu.sg`, then run `ssh YOUR_NUS_ID@vanda.nus.edu.sg` and `cd ~/parallelKmean_reproduction`.

### 8.1 Submit a one-GPU job

The PBS script loads the recorded GCC/Open MPI/CUDA modules and builds inside the allocation.

```bash
ssh YOUR_NUS_ID@vanda.nus.edu.sg
cd ~/parallelKmean_reproduction
mkdir -p pbs/logs results data
bash -n pbs/vanda_cuda.pbs
# Request one A40 GPU and launch one MPI rank.
JOB_ID=$(qsub -l select=1:ncpus=36:mpiprocs=1:ngpus=1:mem=240gb -l walltime=01:00:00 -v 'NP=1,CSV=data/blobs_big.csv,K=128,MAXIT=32,SEED=42,CUDA_ARCH=sm_86' pbs/vanda_cuda.pbs)
printf '%s\n' "$JOB_ID"
```

For the recorded two-GPU configuration, use a new working copy, request `select=1:ncpus=72:mpiprocs=2:ngpus=2:mem=480gb`, and set `NP=2`. These are the observed Vanda allocation sizes; check current site policy if rejected. This recipe targets a single node with one MPI rank per GPU. GPU visibility and rank-to-device mapping must be checked before extending to other allocations or multiple nodes.

### 8.2 Run CUDA interactively inside an allocated GPU session

Build the CUDA target and run one MPI rank on the supplied MNIST dataset:

```bash
make cuda CUDA_ARCH=sm_86
mpirun -np 1 ./bin/kmeans_mpi_cuda data/mnist.csv 10 --label-col 0 --seed 42 --max-iter 10 --fixed-iter
```

Adjust the CUDA architecture to the actual GPU. A successful program prints a `RESULT mpi_cuda ...` line; a scheduler exit code alone is not a numerical validation.

## 9. Monitor and collect results

### 9.1 Watch scheduler state, CPU time, and wall time

In the same shell where `JOB_ID` was assigned:

```bash
watch -d -n 10 "qstat -x -f $JOB_ID | grep -A 2 -E 'job_state =|Exit_status =|resources_used.cput =|resources_used.walltime =|resources_used.cpupercent =|resources_used.mem =|exec_host =|comment ='"
```

Press Ctrl-C to stop monitoring (the job continues). `Q` means queued, `R` running, `F` finished; check `Exit_status = 0` **and** complete output rows. Logs may arrive in `pbs/logs/` only after completion. `resources_used.cput` is aggregate CPU time, not elapsed wall time. Around 1600% CPU usage corresponds to roughly 16 busy CPU cores; it does not measure GPU utilization.

### 9.2 Save the final status and inspect the output

Use these commands after the job finishes:

```bash
qstat -x -f "$JOB_ID" > "results/${JOB_ID}_scheduler.txt"
ls -lt pbs/logs/
cat results/scaling_mpi.csv  # Atlas only
# Cancel only if intended:
# qdel "$JOB_ID"
```

After a successful Atlas run, preserve its CSV under a job-specific filename before another sweep. On your **local** machine, retrieve that working copy's results and logs with `rsync -av YOUR_NUS_ID@atlas9.nus.edu.sg:~/parallelKmean_reproduction/results/ ./downloaded_results/` and the equivalent command for `pbs/logs/`.

## 10. Recorded results

The 100-iteration Atlas study uses 3,000,000 points, 64 features and 128 clusters:

| Completed job | MPI ranks | Serial seconds | MPI seconds |
| --- | ---: | ---: | ---: |
| 1036952.venus01 | 12 | 1704.105486 | 142.843514 |
| 1034257.venus01 | 16 | 1703.059729 | 108.220590 |
| 1037018.venus01 | 16 | 1716.061499 | 106.207582 |
| 1037036.venus01 | 24 | 1728.390679 | 70.887738 |
| 1036751.venus01 | 32 | 1694.752549 | 53.819387 |
| 1036776.venus01 | 48 | 1722.174826 | 36.422506 |

Vanda's completed 32-iteration jobs 1533324 and 1533344 record 10.159337 s on one GPU and 5.073951 s on two GPUs. Earlier completed CPU/GPU series remain in the report and their original CSVs. Local laptop/workstation runs are labelled **Local run**; they do not receive invented scheduler IDs. Historical HPC series lacking a retained ID remain labelled by platform and evidence level.

Speedup is the matched serial time divided by parallel time; efficiency is speedup divided by rank count. Small values above 100% are observed ratios that may reflect cache effects or timing variation, not proof of superlinear scaling. Single-run measurements have no confidence interval. Do not compare the 100-iteration Atlas workload with the 32-iteration GPU workload as a controlled CPU/GPU speedup: their iteration budgets differ. The downloaded Atlas and Vanda inputs now have matching checksums, recorded in pbs/hpc_reconciliation.json; the historical jobs did not record contemporaneous hashes.

The report's evidence manifest identifies the CSVs used in figures. Failed attempts and raw download staging directories are excluded from the submission archive. The report distinguishes retained logs, retained CSVs and session-record evidence; not every historical run has a full archived log or label file.

## 11. Optional analysis and report tools

In a fresh execution copy, after building, `FIXEDITER=1 MAXIT=32 REPS=3 bash scripts/run_scaling.sh data/blobs_big.csv 128 '1 2 4 8 16'` runs serial plus MPI and writes `results/scaling_mpi.csv`. It keeps the minimum time over repetitions. Allocate at least as many CPUs as the largest rank count. Python plotting and report generation are optional and do not run K-means; see [report build instructions](report/BUILD_REPORT.md). The AI acknowledgment and team responsibility statement appear after the report references.
