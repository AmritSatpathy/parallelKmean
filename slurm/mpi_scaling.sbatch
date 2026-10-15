#!/bin/bash
# Strong-scaling run for the MPI (CPU) K-means on Atlas/Vanda via Slurm.
#
# Submit once per process count, e.g.:
#   for p in 1 2 4 8 16 32 64; do sbatch --ntasks=$p slurm/mpi_scaling.sbatch; done
#
# EDIT the module names / partition to match your cluster. Check with
# `module avail` and `sinfo`.
#SBATCH --job-name=kmeans_mpi
#SBATCH --output=slurm/logs/mpi_%j_%N.out
#SBATCH --ntasks=8               # overridden by --ntasks on the sbatch line
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
# #SBATCH --partition=general    # <-- set your partition
# #SBATCH --nodes=1              # <-- let the scheduler place tasks, or pin

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs results

# --- modules (EDIT for Atlas/Vanda) ---
# module purge
# module load gcc openmpi

make mpi gen

# Large synthetic dataset sized so the serial baseline runs ~10 min.
# Generate once (idempotent): 3M points x 64 dims, K=128.
CSV="${CSV:-data/blobs_big.csv}"
K="${K:-128}"
SEED="${SEED:-42}"
MAXIT="${MAXIT:-32}"
if [ ! -s "$CSV" ]; then
    ./bin/gen_blobs --n 3000000 --dims 64 --k "$K" --sep 25 --out "$CSV"
fi

echo "nodes=$SLURM_JOB_NUM_NODES ntasks=$SLURM_NTASKS on $(hostname)"
# --fixed-iter runs exactly MAXIT iterations for a fair, predictable benchmark.
srun --mpi=pmix ./bin/kmeans_mpi "$CSV" "$K" \
     --label-col 0 --seed "$SEED" --max-iter "$MAXIT" --fixed-iter
# If --mpi=pmix is unavailable, try: srun --mpi=pmi2 ...   or   mpirun ...
