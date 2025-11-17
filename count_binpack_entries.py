#!/usr/bin/env python3
from pathlib import Path
import struct
import sys


def count_entries_and_validate(path, detect_errors=True, report_every=1000000):
    count = 0
    errors = []
    with open(path, 'rb') as f:
        size = f.seek(0, 2) or f.tell()
        f.seek(0)
        offset = 0
        while offset + 8 <= size:
            f.seek(offset)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            if hdr[0:4] != b'BINP':
                errors.append((offset, hdr[:4]))
                break
            chunk_size = struct.unpack_from('<I', hdr, 4)[0]
            # read payload
            payload = f.read(chunk_size)
            if len(payload) < chunk_size:
                errors.append((offset, f'payload_mismatch expected={chunk_size} got={len(payload)}'))
                break
            # iterate payload
            off = 0
            while off + 32 + 2 <= len(payload):
                # read packed entry, movelist length
                # take care to not go beyond
                numPlies = (payload[off + 32] << 8) | payload[off + 33]
                off += 32 + 2
                off += numPlies
                count += 1
                if detect_errors and count % report_every == 0:
                    print(f"{path}: counted {count} entries; file offset={offset} payload processed={off}")
            offset += 8 + chunk_size
    return count, errors


if __name__ == '__main__':
    data_dir = Path('data')
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    files = list(data_dir.rglob('*.binpack')) + list(data_dir.rglob('*.no-db.binpack'))
    total = 0
    for f in files:
        print('Checking', f)
        n, errors = count_entries_and_validate(f)
        print(f.name, 'entries:', n, 'errors:', errors[:3])
        total += n
    print('Total entries:', total)
