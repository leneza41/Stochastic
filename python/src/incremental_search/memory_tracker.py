from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import sys

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class MemoryTracker:
    """
    Provides cross-platform process memory measurement (Resident Set Size in MB
    via psutil or native Windows 11 64-bit Win32 K32GetProcessMemoryInfo API)
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
                pass

        if sys.platform == "win32":
            try:
                k32 = ctypes.windll.kernel32
                k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
                k32.K32GetProcessMemoryInfo.argtypes = [
                    ctypes.wintypes.HANDLE,
                    ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                    ctypes.wintypes.DWORD,
                ]
                k32.K32GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
                pmc = _PROCESS_MEMORY_COUNTERS(
                    cb=ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
                )
                if k32.K32GetProcessMemoryInfo(
                    k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
                ):
                    return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
            except Exception:
                pass

        return 0.0
