#!/usr/bin/env python3
from pathlib import Path
import sys
from statistics import mean, median

if __name__ == '__main__':
    data_dir = Path('data')
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    files = list(data_dir.rglob('*.binpack')) + list(data_dir.rglob('*.no-db.binpack'))
    sizes = [f.stat().st_size for f in files]
    print('files:', len(files))
    if sizes:
        print('total bytes', sum(sizes), 'total GiB:', sum(sizes)/(1024**3))
        print('mean bytes', mean(sizes))
        print('median bytes', median(sizes))
        print('small files <1k:', sum(1 for s in sizes if s<1024))
        print('small files <1MB:', sum(1 for s in sizes if s<1024*1024))
        print('small files <10MB:', sum(1 for s in sizes if s<10*1024*1024))
        print('top 5 largest sizes:', sorted(sizes, reverse=True)[:5])
        print('bottom 5 smallest sizes:', sorted(sizes)[:5])
    else:
        print('no binpack files found in', data_dir)
