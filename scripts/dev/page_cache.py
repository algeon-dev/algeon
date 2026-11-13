"""Page-cache residency, eviction and memory counters for benchmark drivers.

A benchmark that reads Parquet files from the OS page cache records how much of each file
was resident before and after the run. `evict` drops the clean cached pages of the files of
other datasets so that a large dataset does not compete with a previous one for memory.
"""
import ctypes
import mmap
import os
from pathlib import Path


def residency(data_dir):
    """Resident fraction of every *.parquet file in `data_dir`, from mincore on a read-only mapping."""
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    page = mmap.PAGESIZE
    result = {}
    for f in sorted(Path(data_dir).glob('*.parquet')):
        size = f.stat().st_size
        fd = os.open(f, os.O_RDONLY)
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        pages = (size + page - 1) // page
        vec = (ctypes.c_ubyte * pages)()
        rc = libc.mincore(ctypes.c_void_p(addr), size, vec)
        res = bytes(b & 1 for b in bytes(vec)).count(1) if rc == 0 else None
        libc.munmap(ctypes.c_void_p(addr), size)
        os.close(fd)
        result[f.name] = {'bytes': size, 'resident_pages': res, 'pages': pages,
                          'resident_fraction': res / pages if res is not None else None}
    total_b = sum(v['bytes'] for v in result.values())
    total_res = sum((v['resident_fraction'] or 0) * v['bytes'] for v in result.values())
    return {'files': result, 'total_bytes': total_b, 'resident_fraction_by_bytes': total_res / total_b}


def evict(data_dir):
    """Drop the clean cached pages of every *.parquet file in `data_dir` with POSIX_FADV_DONTNEED."""
    for f in sorted(Path(data_dir).glob('*.parquet')):
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def meminfo():
    out = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, v = line.split(':', 1)
        if k in ('MemTotal', 'MemFree', 'MemAvailable', 'Cached', 'Buffers'):
            out[k] = v.strip()
    return out
