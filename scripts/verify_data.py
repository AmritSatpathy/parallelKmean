#!/usr/bin/env python3
"""Verify downloaded/generated datasets against the recorded project inputs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--files', nargs='+', default=['mnist_train.csv', 'mnist_test.csv', 'mnist.csv'])
    parser.add_argument('--path', type=Path, help='Check a temporary download; requires exactly one --files name')
    args = parser.parse_args()
    expected = {e['file']: e for e in json.loads((ROOT / 'data/checksums.json').read_text())['files']}
    if args.path and len(args.files) != 1:
        parser.error('--path requires exactly one --files name')
    for name in args.files:
        if name not in expected:
            parser.error('Unknown dataset: ' + name)
        path = args.path or args.data_dir / name
        if not path.is_file():
            raise SystemExit('Missing dataset: ' + str(path))
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        if path.stat().st_size != expected[name]['bytes'] or h.hexdigest() != expected[name]['sha256']:
            raise SystemExit(f'CHECKSUM MISMATCH: {path}\nExpected: {expected[name]["sha256"]}\nActual:   {h.hexdigest()}')
        print(f'PASS {name}: {h.hexdigest()}')


if __name__ == '__main__':
    main()
