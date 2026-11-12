#!/usr/bin/env bash
# Strong-scaling sweep for the MPI (CPU) K-means, plus the serial baseline.
# Emits results/scaling_mpi.csv with columns:
#   impl,nproc,n,dims,k,iters,inertia,time
#
# Strong scaling: fixed problem size, increasing process count. Speedup S(p) =
# T_serial / T(p); efficiency E(p) = S(p)/p.
#
# Usage:
#   bash scripts/run_scaling.sh data/mnist.csv 10 "1 2 4 8 16"
set -euo pipefail
cd "$(dirname "$0")/.."

CSV="${1:-data/mnist.csv}"
K="${2:-10}"
PROCS="${3:-1 2 4 8 16}"
SEED="${SEED:-42}"
MAXIT="${MAXIT:-50}"
REPS="${REPS:-3}"          # repeats; we keep the best (min) time
LABELCOL="${LABELCOL:-0}"
# FIXEDITER=1 forces exactly MAXIT iterations (predictable, fair benchmark
# across process counts). Recommended for scaling studies.
FIXEDITER="${FIXEDITER:-0}"
FIXED_FLAG=""
[ "$FIXEDITER" = "1" ] && FIXED_FLAG="--fixed-iter"

mkdir -p results
OUT=results/scaling_mpi.csv
echo "impl,nproc,n,dims,k,iters,inertia,time" > "$OUT"

# Pull the "RESULT ..." line and reformat into a CSV row.
emit() {  # $1=impl $2=nproc  reads RESULT line from stdin
    awk -v impl="$1" -v p="$2" '
        /^RESULT/ {
            for (i=1;i<=NF;i++){ split($i,kv,"="); v[kv[1]]=kv[2] }
            print impl","p","v["n"]","v["dims"]","v["k"]","v["iters"]","v["inertia"]","v["time"]
        }'
}

best_time=""
run_best() {  # runs REPS times, echoes the RESULT line with the min time
    local line="" t best_line="" best_t=""
    for _ in $(seq "$REPS"); do
        line="$("$@" 2>/dev/null | grep '^RESULT' || true)"
        [ -z "$line" ] && continue
        t="$(echo "$line" | sed -n 's/.*time=\([0-9.]*\).*/\1/p')"
        if [ -z "$best_t" ] || awk -v a="$t" -v b="$best_t" 'BEGIN{exit !(a<b)}'; then
            best_t="$t"; best_line="$line"
        fi
    done
    echo "$best_line"
}

echo "=== serial baseline ==="
run_best ./bin/kmeans_serial "$CSV" "$K" --label-col "$LABELCOL" --seed "$SEED" \
    --max-iter "$MAXIT" $FIXED_FLAG \
    | emit serial 1 | tee -a "$OUT"

for p in $PROCS; do
    echo "=== mpi np=$p ==="
    run_best mpirun --oversubscribe -np "$p" ./bin/kmeans_mpi "$CSV" "$K" \
        --label-col "$LABELCOL" --seed "$SEED" --max-iter "$MAXIT" $FIXED_FLAG \
        | emit mpi "$p" | tee -a "$OUT"
done

echo
echo "wrote $OUT"
cat "$OUT"
