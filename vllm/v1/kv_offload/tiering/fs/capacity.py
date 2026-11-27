# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRU capacity manager for the file system KV offload tier.

The fs tier has no inherent size bound: every offloaded chunk becomes a file
and nothing ever deletes them. FsCapacityManager tracks the tier's chunk files
in an LRU ordering and, once total size exceeds ``capacity_bytes``, unlinks
least-recently-used files until usage drops below
``capacity_bytes * watermark``.

Concurrency model: all mutating calls happen on the scheduler thread (the
tier's ``submit_store`` / ``touch`` / ``get_finished_jobs`` are only invoked
from there), so a plain dict suffices; a lock is kept for safety against
future callers. Actual ``unlink`` latency is metadata-only and small, and each
call to :meth:`evict` is bounded by ``max_evictions_per_step``.

Load-race protection: a file can be looked up (HIT) and then evicted before
the load job runs, which would fail the load. Files whose keys were recently
touched or stored are protected from eviction for ``protect_s`` seconds.

Restart behavior: the index is rebuilt by scanning ``scan_dirs`` at startup,
seeding recency from file mtimes, so the LRU ordering survives restarts
approximately.
"""

import os
import threading
import time
from collections import OrderedDict

from vllm.logger import init_logger

logger = init_logger(__name__)


class FsCapacityManager:
    def __init__(
        self,
        capacity_bytes: int,
        watermark: float = 0.9,
        protect_s: float = 120.0,
        max_evictions_per_step: int = 1024,
    ) -> None:
        assert capacity_bytes > 0
        assert 0.0 < watermark <= 1.0
        self.capacity_bytes = capacity_bytes
        self.low_bytes = int(capacity_bytes * watermark)
        self.protect_s = protect_s
        self.max_evictions_per_step = max_evictions_per_step

        # path -> (size_bytes, last_use_monotonic); insertion order == LRU
        self._files: OrderedDict[str, tuple[int, float]] = OrderedDict()
        self.total_bytes = 0
        self.evicted_files_total = 0
        self.evicted_bytes_total = 0
        self._lock = threading.Lock()

    def scan(self, scan_dirs: list[str]) -> None:
        """Rebuild the index from disk, ordering recency by file mtime."""
        t0 = time.perf_counter()
        entries: list[tuple[float, str, int]] = []
        for scan_dir in scan_dirs:
            if not os.path.isdir(scan_dir):
                continue
            for dirpath, _, filenames in os.walk(scan_dir):
                for name in filenames:
                    if not name.endswith(".bin"):
                        continue
                    path = os.path.join(dirpath, name)
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    entries.append((st.st_mtime, path, st.st_size))
        entries.sort()
        # Seed recency outside the protection window so files surviving from a
        # previous run are immediately evictable if we start over capacity.
        stale = time.monotonic() - self.protect_s - 1.0
        with self._lock:
            for _, path, size in entries:
                self._files[path] = (size, stale)
                self.total_bytes += size
        logger.info(
            "fs tier capacity index: %d files, %.2f GB (capacity %.2f GB), "
            "scanned in %.1fs",
            len(entries),
            self.total_bytes / 1e9,
            self.capacity_bytes / 1e9,
            time.perf_counter() - t0,
        )

    def record_store(self, path: str, size: int) -> None:
        """Account a newly stored (or overwritten) file as most recently used."""
        now = time.monotonic()
        with self._lock:
            old = self._files.pop(path, None)
            if old is not None:
                self.total_bytes -= old[0]
            self._files[path] = (size, now)
            self.total_bytes += size
            self._n_stores = getattr(self, "_n_stores", 0) + 1
            if self._n_stores % 5000 == 0:
                logger.info(
                    "fs tier capacity: %d stores accounted, usage %.2f/%.2f GB",
                    self._n_stores,
                    self.total_bytes / 1e9,
                    self.capacity_bytes / 1e9,
                )

    def record_use(self, path: str) -> None:
        """Mark a file most recently used and protect it from eviction."""
        now = time.monotonic()
        with self._lock:
            entry = self._files.pop(path, None)
            if entry is None:
                return
            self._files[path] = (entry[0], now)

    def evict(self) -> int:
        """Evict LRU files until usage <= low watermark. Returns bytes freed."""
        if self.total_bytes <= self.capacity_bytes:
            return 0
        freed = 0
        n_files = 0
        now = time.monotonic()
        with self._lock:
            for _ in range(self.max_evictions_per_step):
                if self.total_bytes - freed <= self.low_bytes:
                    break
                if not self._files:
                    break
                path, (size, last_use) = next(iter(self._files.items()))
                if now - last_use < self.protect_s:
                    # LRU head is inside the protection window, so every other
                    # entry is too (younger last_use). Nothing evictable now.
                    break
                self._files.popitem(last=False)
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.warning("fs tier eviction: unlink(%s) failed: %s", path, e)
                freed += size
                n_files += 1
            self.total_bytes -= freed
            self.evicted_files_total += n_files
            self.evicted_bytes_total += freed
        if n_files:
            logger.info(
                "fs tier evicted %d files (%.2f GB); usage %.2f/%.2f GB",
                n_files,
                freed / 1e9,
                self.total_bytes / 1e9,
                self.capacity_bytes / 1e9,
            )
        return freed
