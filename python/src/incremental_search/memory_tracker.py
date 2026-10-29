from __future__ import annotations

import os

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


class MemoryTracker:
    """
    Provides process memory measurement (Resident Set Size in MB via psutil)
    and converts byte counts of algorithm-specific data structures into MB.
    """

    @staticmethod
    def bytes_to_mb(num_bytes: int) -> float:
        return float(num_bytes) / (1024.0 * 1024.0)

    @staticmethod
    def current_process_rss_mb() -> float:
        if psutil is not None:
            try:
                process = psutil.Process(os.getpid())
                return float(process.memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                return 0.0
        return 0.0
