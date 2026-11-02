#!/usr/bin/env python3
"""Download pinned public inputs and reject any checksum mismatch (stdlib only)."""
import argparse
import gzip
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, required=True)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((here / 'source_manifest.json').read_text())
    for entry in manifest['files']:
        dest = args.raw_dir / entry['local_name']
        if dest.exists() and sha256(dest) == entry['sha256']:
            print('Verified cache:', dest.name)
            continue
        temp = dest.with_suffix(dest.suffix + '.part')
        snapshot = here / 'source_snapshots' / (dest.name + '.gz')
        if snapshot.exists():
            with gzip.open(snapshot, 'rb') as r, temp.open('wb') as w:
                shutil.copyfileobj(r, w)
        else:
            with urllib.request.urlopen(entry['url'], timeout=120) as r, temp.open('wb') as w:
                shutil.copyfileobj(r, w)
        if sha256(temp) != entry['sha256']:
            temp.unlink()
            raise RuntimeError('Source has changed; refusing to substitute new data: ' + dest.name)
        temp.replace(dest)
        print('Downloaded and verified:', dest.name)


if __name__ == '__main__':
    main()
