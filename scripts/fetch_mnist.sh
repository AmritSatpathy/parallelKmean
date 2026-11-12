#!/usr/bin/env bash
# Fetch the exact MNIST CSV inputs verified against the project on 2026-11-12.
# Usage: bash scripts/fetch_mnist.sh
# Optional: DATA_DIR=/absolute/path bash scripts/fetch_mnist.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${DATA_DIR:-$PWD/data}"
mkdir -p "$DATA_DIR"
command -v python3 >/dev/null
TEMP_FILE=""
trap 'if [ -n "$TEMP_FILE" ]; then rm -f "$TEMP_FILE"; fi' EXIT

check() {
    python3 scripts/verify_data.py --files "$1" --path "$2"
}
fetch() {
    local name="$1" out="$DATA_DIR/$1" base
    if [ -s "$out" ]; then
        check "$name" "$out"
        return
    fi
    TEMP_FILE=$(mktemp "$DATA_DIR/.${name}.XXXXXX")
    for base in https://data.pjreddie.com/files https://raw.githubusercontent.com/halimb/MNIST-csv/master; do
        echo "Downloading $base/$name"
        if command -v curl >/dev/null; then
            if ! curl -fL --retry 3 --connect-timeout 20 -o "$TEMP_FILE" "$base/$name"; then continue; fi
        else
            if ! wget -O "$TEMP_FILE" "$base/$name"; then continue; fi
        fi
        if check "$name" "$TEMP_FILE"; then
            mv "$TEMP_FILE" "$out"
            TEMP_FILE=""
            return
        fi
    done
    echo "Could not obtain the verified $name from either source." >&2
    return 1
}
fetch mnist_train.csv
fetch mnist_test.csv
TEMP_FILE=$(mktemp "$DATA_DIR/.mnist.csv.XXXXXX")
cat "$DATA_DIR/mnist_train.csv" "$DATA_DIR/mnist_test.csv" > "$TEMP_FILE"
check mnist.csv "$TEMP_FILE"
mv "$TEMP_FILE" "$DATA_DIR/mnist.csv"
TEMP_FILE=""
echo "Ready: $DATA_DIR/mnist.csv (70000 rows; label + 784 features)"
