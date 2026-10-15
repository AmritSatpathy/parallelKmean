#!/bin/bash
# MPI+CUDA K-means on Atlas/Vanda via Slurm.
#
# One MPI rank per GPU is the intended mapping. Request GPUs with --gres and
# match --ntasks to the total GPU count across the requested nodes.
#
# Example (2 nodes x 2 GPUs = 4 ranks):
#   sbatch --nodes=2 --ntasks=4 --gres=gpu:2 slurm/cuda.sbatch
#
# EDIT module names, partition, and CUDA_ARCH for your cluster's GPUs.
#SBATCH --job-name=kmeans_cuda
#SBATCH --output=slurm/logs/cuda_%j_%N.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
# #SBATCH --partition=gpu       # <-- set your GPU partition

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs results

# --- modules (EDIT for Atlas/Vanda) ---
# module purge
# module load gcc openmpi cuda
# CUDA_ARCH must match the GPU: V100=sm_70, A100=sm_80, H100=sm_90.
ARCH="${CUDA_ARCH:-sm_80}"

make cuda gen CUDA_ARCH="$ARCH"

# Large synthetic dataset (3M x 64, K=128), generated once if absent.
CSV="${CSV:-data/blobs_big.csv}"
K="${K:-128}"
SEED="${SEED:-42}"
MAXIT="${MAXIT:-32}"
if [ ! -s "$CSV" ]; then
    ./bin/gen_blobs --n 3000000 --dims 64 --k "$K" --sep 25 --out "$CSV"
fi

echo "nodes=$SLURM_JOB_NUM_NODES ntasks=$SLURM_NTASKS gres=$SLURM_JOB_GPUS on $(hostname)"
srun --mpi=pmix ./bin/kmeans_mpi_cuda "$CSV" "$K" \
     --label-col 0 --seed "$SEED" --max-iter "$MAXIT" --fixed-iter
