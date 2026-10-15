#!/usr/bin/env bash
# Download MNIST in CSV form (label in column 0, 784 pixel features after).
# Primary source: pjreddie mirror. Falls back to a GitHub mirror if needed.
#
# Output:
#   data/mnist_train.csv  (60000 rows)
#   data/mnist_test.csv   (10000 rows)
#   data/mnist.csv        (train+test concatenated, 70000 rows) -- used for runs
#
# Usage: bash scripts/fetch_mnist.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

fetch() {
    local url="$1" out="$2"
    if [ -s "$out" ]; then echo "have $out"; return 0; fi
    echo "downloading $url"
    if command -v curl >/dev/null; then
        curl -fL --retry 3 -o "$out" "$url"
    else
        wget -q -O "$out" "$url"
    fi
}

PRI="https://pjreddie.com/media/files"
ALT="https://raw.githubusercontent.com/halimb/MNIST-csv/master"

fetch "$PRI/mnist_train.csv" data/mnist_train.csv || \
    fetch "$ALT/mnist_train.csv" data/mnist_train.csv
fetch "$PRI/mnist_test.csv" data/mnist_test.csv || \
    fetch "$ALT/mnist_test.csv" data/mnist_test.csv

cat data/mnist_train.csv data/mnist_test.csv > data/mnist.csv
echo "rows: $(wc -l < data/mnist.csv) in data/mnist.csv"
echo "cols: $(head -1 data/mnist.csv | awk -F, '{print NF}')"
