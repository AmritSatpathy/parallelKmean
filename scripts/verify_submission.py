#!/usr/bin/env python3
"""Check an extracted submission against its packaged SHA-256 manifest.

Run: python3 scripts/verify_submission.py
This checks files only; it does not build, run K-means or contact an HPC system.
"""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / 'SUBMISSION_MANIFEST.json'
    if not path.is_file():
        raise SystemExit('Run this inside the extracted submission ZIP; its manifest is generated during packaging.')
    manifest = json.loads(path.read_text())
    failures = []
    for item in manifest['files']:
        source = root / item['path']
        if not source.is_file():
            failures.append(item['path'] + ': missing')
            continue
        digest = hashlib.sha256()
        with source.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        if source.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            failures.append(item['path'] + ': changed')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"PASS: {len(manifest['files'])} packaged files match their SHA-256 checksums.")
    print('This verifies file integrity, not benchmark performance or numerical correctness.')


if __name__ == '__main__':
    main()
