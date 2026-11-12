# Parallel K-means with MPI and CUDA

DSA5208 · Scalable Distributed Computing for Data Science · NUS · November 2026

| Team member | Student ID |
| --- | --- |
| Amrit Satpathy | A0355181E |
| Ritik Srivastava | A0355860X |

Repository: <https://github.com/AmritSatpathy/parallelKmean>
Canonical branch: [`mainline`](https://github.com/AmritSatpathy/parallelKmean/tree/mainline). The submission ZIP is a frozen source snapshot with the run documentation and checksum tools; dataset CSVs are downloaded or generated separately.

After extracting the submission ZIP, run `python3 scripts/verify_submission.py` to check all packaged file checksums (no computation is performed).

Start with [the final PDF report](report/Parallel_Kmeans_Project_Report.pdf). It explains Lloyd's algorithm, MPI decomposition and collectives, CUDA kernels, validation, scaling, MNIST interpretation, limitations, references, and AI usage. All results are one team study. Only completed experiments are used in the report.

## Table of contents

1. [Where to start](#where-to-start)
2. [Get the code and download data](#get-the-code-and-download-data)
3. [Contents and evidence](#contents-and-evidence)
4. [Prerequisites](#prerequisites)
5. [Local CPU quick start](#local-cpu-quick-start)
6. [MNIST and synthetic workloads](#mnist-and-synthetic-workloads)
7. [Copy and synchronize code and data to HPC](#copy-and-synchronize-code-and-data-to-hpc)
8. [Atlas: reproduce in a fresh job directory](#atlas-reproduce-in-a-fresh-job-directory)
9. [Vanda: one or two GPUs](#vanda-one-or-two-gpus)
10. [Monitor and retain a job](#monitor-and-retain-a-job)
11. [Reported results and interpretation](#reported-results-and-interpretation)
12. [Optional scaling and report tools](#optional-scaling-and-report-tools)

## Where to start

| Your goal | Follow these sections |
| --- | --- |
| Review the submission without executing anything | Final PDF → Contents and evidence → Reported results |
| Try a small CPU example | Get the code/data → Prerequisites → Local CPU quick start |
| Run on Atlas | Get the code/data → Copy/sync → Atlas submission → Monitor/download |
| Run on Vanda GPUs | Get the code/data → Copy/sync (choose Vanda) → Vanda submission → Monitor/download |
| Inspect or regenerate report figures | Reported results → Optional tools → report/BUILD_REPORT.md |

Commands are labelled by where they run: **local computer** or **HPC**. Copy only the commands inside code blocks, not a shell prompt or example output. Replace `YOUR_NUS_ID` with your login. Use a fresh working copy for each benchmark; the supplied `results/` files are historical evidence.

## Get the code and download data

Follow these steps in order. The Canvas ZIP omits **all dataset CSVs** to keep the upload small; source, completed-run results, logs and data checksums remain included.

**1. Get the code.** Either extract the submitted code ZIP and enter its `parallelKmean` folder (recommended: this is the exact submitted version), or clone the repository:

```bash
git clone --branch mainline https://github.com/AmritSatpathy/parallelKmean.git
cd parallelKmean
```

The ZIP is authoritative for the submitted version; `mainline` is the canonical repository branch. The updated workflow below is available on GitHub once the submission-update pull request is merged. Until then, use the submitted ZIP.

**2. Download MNIST into `data/`.** Requires Python 3 and curl (or wget), plus internet access. Run from the project root:

```bash
bash scripts/fetch_mnist.sh
python3 scripts/verify_data.py
```

This downloads `mnist_train.csv` and `mnist_test.csv`, verifies their SHA-256 checksums, then concatenates them into `data/mnist.csv` and checks that file too. The public files were freshly downloaded and confirmed byte-identical to the project inputs on 12 November 2026. Existing files are verified before reuse; a failed download does not become a finished CSV.

Sources: [MNIST-in-CSV page](https://pjreddie.com/projects/mnist-in-csv/), [training CSV](https://data.pjreddie.com/files/mnist_train.csv), [test CSV](https://data.pjreddie.com/files/mnist_test.csv). The script has a [GitHub fallback](https://github.com/halimb/MNIST-csv), accepted only if the same checksums pass. No public download exists for this project's generated `blobs_big.csv`.

**3. Build and run.** On a local CPU machine, follow Prerequisites and Local CPU quick start below. For MNIST specifically, the complete CPU sequence after downloading is:

```bash
make serial mpi gen
mkdir -p results/reproduction
./bin/kmeans_serial data/mnist.csv 10 --label-col 0 --seed 42 --max-iter 32 --fixed-iter
mpirun -np 2 ./bin/kmeans_mpi data/mnist.csv 10 --label-col 0 --seed 42 --max-iter 32 --fixed-iter
```

On Atlas/Vanda, instead follow Copy/sync and the relevant PBS submission section; compilation and computation take place inside the allocation. MNIST can be downloaded on your local computer and synced if the cluster cannot access the public server.

**4. For the large synthetic benchmark**, use the generator instructions below or submit the documented PBS job: it generates `data/blobs_big.csv` if missing. Synthetic generation recreates the workload, but byte-for-byte agreement can depend on the compiler/C++ library. Compare its checksum against the archived input before claiming exact-input reproduction. No synthetic dataset is disguised as an external download.

## Contents and evidence

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

The ZIP and Git repository exclude dataset CSVs. Follow the download/generation sequence above. See [data notes](data/README.md) for verified sources and checksums. No precompiled binaries are supplied: compile for the execution machine.

## Prerequisites

CPU: a C++17 compiler, GNU Make, Bash and Open MPI (`mpicxx`, `mpirun`). The Atlas runs used GCC 7.3.0 and Open MPI 4.1.1. CUDA additionally requires an NVIDIA GPU, a compatible driver and CUDA toolkit; the Vanda configuration used GCC 13.2.0, Open MPI 4.1.6 and CUDA 12.4.0 with A40 GPUs (`sm_86`). Python 3 is needed for validation; plotting requires matplotlib. CUDA is optional for the CPU implementation. `make all` requires CUDA; use the explicit CPU targets below on CPU machines.

## Local CPU quick start

Run these commands from an extracted copy of the ZIP, inside `parallelKmean`. Use a fresh extraction for experiments so archived results remain intact. On HPC, run computations inside a scheduler allocation, not on the login node.

```bash
make serial mpi gen
mkdir -p data results/reproduction
./bin/gen_blobs --n 10000 --dims 16 --k 8 --seed 0 --sep 25 --out data/smoke.csv
./bin/kmeans_serial data/smoke.csv 8 --label-col 0 --seed 42 --max-iter 20 --fixed-iter --out-labels results/reproduction/serial.labels
mpirun -np 2 ./bin/kmeans_mpi data/smoke.csv 8 --label-col 0 --seed 42 --max-iter 20 --fixed-iter --out-labels results/reproduction/mpi.labels
python3 scripts/compare_labels.py results/reproduction/serial.labels results/reproduction/mpi.labels
python3 scripts/validate.py --pred results/reproduction/mpi.labels --csv data/smoke.csv --truth-col 0
```

`compare_labels.py` reports raw and majority-mapped agreement; the latter is a heuristic, not an optimal one-to-one label permutation. Compare equal-length label files. Floating-point reduction order can affect labels near boundaries, so matching displayed inertia alone is not proof of identical clustering.

The default compiler flags contain `-march=native`. Build on the execution machine, or use `make serial mpi gen CXXFLAGS='-O3 -std=c++17 -Wall -Wextra'` for a portable CPU build.

## MNIST and synthetic workloads

MNIST has 70,000 rows, with the ground-truth digit in column 0 and 784 pixel features. Always pass `--label-col 0` so the digit is excluded from clustering features. For a real-data run in a suitable allocation:

```bash
test -s data/mnist.csv
./bin/kmeans_serial data/mnist.csv 10 --label-col 0 --seed 42 --max-iter 100 --out-labels results/reproduction/mnist.labels
python3 scripts/validate.py --pred results/reproduction/mnist.labels --csv data/mnist.csv --truth-col 0
```

Omitting `--fixed-iter` permits early stopping. The synthetic CSV is excluded from the ZIP. After `make gen`, generate the 3,000,000-row workload on an allocated compute node (or a suitable local machine):

```bash
./bin/gen_blobs --n 3000000 --dims 64 --k 128 --spread 1 --sep 25 --seed 0 --out data/blobs_big.csv
```

Generation seed 0 and clustering seed 42 are different parameters. On 12 November, the downloaded Atlas and Vanda `blobs_big.csv` files were checked: each is 1,757,795,622 bytes, 3,000,000 rows and 65 columns, with identical SHA-256 `4c420e3b2e6b442e807a658dd8d0c3d9b1cfb685e51c8868863acc51e8d3de2b`. The large CSV is deliberately omitted from the ZIP. This verifies the downloaded datasets; the older job logs did not capture contemporaneous input hashes. Check a generated file with `python3 scripts/verify_data.py --files blobs_big.csv`. If its checksum differs, it is a new workload instance: keep its results separate from the archived measurements. The generator uses C++ random distributions, whose output may differ between standard-library implementations.

## Copy and synchronize code and data to HPC

Run this block on your **local computer**, from the extracted `parallelKmean` folder. Connect to the NUS VPN first. Set the account and choose Atlas or Vanda. The final slash on `./` means copy the folder contents.

```bash
HPC_USER=YOUR_NUS_ID
HPC_HOST=atlas9.nus.edu.sg
# For Vanda instead: HPC_HOST=vanda.nus.edu.sg
REMOTE_DIR=parallelKmean_reproduction
ssh "$HPC_USER@$HPC_HOST" "mkdir -p ~/$REMOTE_DIR"
rsync -av --progress \
  --exclude '.git/' --exclude 'bin/' --exclude '.venv/' \
  --exclude '__pycache__/' --exclude '.DS_Store' \
  --exclude 'results/' --exclude 'pbs/logs/' \
  ./ "$HPC_USER@$HPC_HOST:~/$REMOTE_DIR/"
```

This copies source, scripts, documentation and **all files in `data/`**, while leaving remote benchmark outputs alone. Repeat the same `rsync` command for incremental sync, only when no job is using that working directory. Do not sync code over a running job; choose a new `REMOTE_DIR` for a new experiment. There is no `--delete`, so extra remote files are preserved.

To sync only the dataset, retaining exactly the local file contents:

```bash
rsync -av --progress data/ "$HPC_USER@$HPC_HOST:~/$REMOTE_DIR/data/"
shasum -a 256 data/mnist.csv
ssh "$HPC_USER@$HPC_HOST" "sha256sum ~/$REMOTE_DIR/data/mnist.csv"
```

The two hashes should match. For a generated or retained synthetic file on your local machine, copy it instead of regenerating it on the cluster:

```bash
# Only if this file is available locally:
test -s data/blobs_big.csv && rsync -av --progress data/blobs_big.csv "$HPC_USER@$HPC_HOST:~/$REMOTE_DIR/data/"
# Compare local shasum and remote sha256sum for blobs_big.csv as above.
```

If the large synthetic dataset is already on the **same HPC system** in `~/parallelKmean/data/`, copy it into the fresh working directory there:

```bash
# Run after SSH, on the HPC system:
mkdir -p ~/parallelKmean_reproduction/data
cp -p ~/parallelKmean/data/blobs_big.csv ~/parallelKmean_reproduction/data/blobs_big.csv
sha256sum ~/parallelKmean/data/blobs_big.csv ~/parallelKmean_reproduction/data/blobs_big.csv
```

This preserves the original HPC dataset. If it is unavailable, use the documented generator and label the resulting measurements as a new reproduction.

After completion, download results from your **local** terminal. Use a new destination each time to preserve previous runs:

```bash
DOWNLOAD_DIR="hpc_download_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DOWNLOAD_DIR/results" "$DOWNLOAD_DIR/logs"
rsync -av "$HPC_USER@$HPC_HOST:~/$REMOTE_DIR/results/" "$DOWNLOAD_DIR/results/"
rsync -av "$HPC_USER@$HPC_HOST:~/$REMOTE_DIR/pbs/logs/" "$DOWNLOAD_DIR/logs/"
```

## Atlas: reproduce in a fresh job directory

Connect to the NUS VPN first. First complete the copy/sync section with `HPC_HOST=atlas9.nus.edu.sg`. A local `/Users/...` path does not exist on Atlas. Then SSH from your local terminal and submit from the remote copy:

```bash
ssh YOUR_NUS_ID@atlas9.nus.edu.sg
cd ~/parallelKmean_reproduction
mkdir -p pbs/logs results data
bash -n pbs/atlas_sweep.pbs
JOB_ID=$(qsub -q parallel -l select=1:ncpus=16:mpiprocs=16:mem=40gb -l walltime=02:00:00 -v 'CSV=data/blobs_big.csv,K=128,PROCS=16,MAXIT=100,REPS=1,SEED=42' pbs/atlas_sweep.pbs)
printf '%s\n' "$JOB_ID"
```

The job builds on the compute node, generates `data/blobs_big.csv` if absent, and runs a serial baseline followed by 16 MPI ranks. For 12, 24, 32 or 48 ranks, change **both** `ncpus`/`mpiprocs` and `PROCS`; use a fresh working copy for each job. The recorded parallel queue minimum was 12 CPUs. Queue policies and available nodes are site-specific. Memory belongs inside the `select` request on this system. The packaged script defaults to a 16-rank job, matching this example.

For MNIST, check the file exists and change the `-v` argument to `CSV=data/mnist.csv,K=10,PROCS=16,MAXIT=32,REPS=1,SEED=42`. Missing real-data inputs must not be replaced with synthetic data.

The Atlas script pins the compiler and MPI paths used in the successful runs. If these paths change, edit the environment block to the matching installed toolchain. Do not mix MVAPICH and Open MPI libraries. Each sweep overwrites `results/scaling_mpi.csv` and `results/scaling_atlas.csv` in its working directory. A full 100-iteration serial baseline takes around 28 minutes in the recorded runs; job wall time includes loading, initialization, compilation and all runs, not just the reported MPI iteration time.

## Vanda: one or two GPUs

Complete the copy/sync section with `HPC_HOST=vanda.nus.edu.sg`, then run `ssh YOUR_NUS_ID@vanda.nus.edu.sg` and `cd ~/parallelKmean_reproduction`. The checked-in PBS file loads the recorded GCC/Open MPI/CUDA modules and builds inside the allocation.

```bash
mkdir -p pbs/logs results data
bash -n pbs/vanda_cuda.pbs
JOB_ID=$(qsub -l select=1:ncpus=36:mpiprocs=1:ngpus=1:mem=240gb -l walltime=01:00:00 -v 'NP=1,CSV=data/blobs_big.csv,K=128,MAXIT=32,SEED=42,CUDA_ARCH=sm_86' pbs/vanda_cuda.pbs)
printf '%s\n' "$JOB_ID"
```

For the recorded two-GPU configuration, use a new working copy, request `select=1:ncpus=72:mpiprocs=2:ngpus=2:mem=480gb`, and set `NP=2`. These are the observed Vanda allocation sizes; check current site policy if rejected. This recipe targets a single node with one MPI rank per GPU. GPU visibility and rank-to-device mapping must be checked before extending to other allocations or multiple nodes.

For local CUDA execution on an allocated GPU machine:

```bash
make cuda CUDA_ARCH=sm_86
mpirun -np 1 ./bin/kmeans_mpi_cuda data/mnist.csv 10 --label-col 0 --seed 42 --max-iter 10 --fixed-iter
```

Adjust the CUDA architecture to the actual GPU. A successful program prints a `RESULT mpi_cuda ...` line; a scheduler exit code alone is not a numerical validation.

## Monitor and retain a job

In the same shell where `JOB_ID` was assigned:

```bash
watch -d -n 10 "qstat -x -f $JOB_ID | grep -A 2 -E 'job_state =|Exit_status =|resources_used.cput =|resources_used.walltime =|resources_used.cpupercent =|resources_used.mem =|exec_host =|comment ='"
```

Press Ctrl-C to stop monitoring (the job continues). `Q` means queued, `R` running, `F` finished; check `Exit_status = 0` **and** complete output rows. Logs may arrive in `pbs/logs/` only after completion. `resources_used.cput` is aggregate CPU time, not elapsed wall time. Around 1600% CPU usage corresponds to roughly 16 busy CPU cores; it does not measure GPU utilization.

```bash
qstat -x -f "$JOB_ID" > "results/${JOB_ID}_scheduler.txt"
ls -lt pbs/logs/
cat results/scaling_mpi.csv  # Atlas only
# Cancel only if intended:
# qdel "$JOB_ID"
```

After a successful Atlas run, preserve its CSV under a job-specific filename before another sweep. On your **local** machine, retrieve that working copy's results and logs with `rsync -av YOUR_NUS_ID@atlas9.nus.edu.sg:~/parallelKmean_reproduction/results/ ./downloaded_results/` and the equivalent command for `pbs/logs/`.

## Reported results and interpretation

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

## Optional scaling and report tools

In a fresh execution copy, after building, `FIXEDITER=1 MAXIT=32 REPS=3 bash scripts/run_scaling.sh data/blobs_big.csv 128 '1 2 4 8 16'` runs serial plus MPI and writes `results/scaling_mpi.csv`. It keeps the minimum time over repetitions. Allocate at least as many CPUs as the largest rank count. Python plotting and report generation are optional and do not run K-means; see [report build instructions](report/BUILD_REPORT.md). The AI acknowledgment and team responsibility statement are in the report's references.
