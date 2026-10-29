#pragma once

#include <cstddef>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace incremental_search {

/**
 * Provides cross-platform process memory measurement (Resident Set Size in MB)
 * and converts byte counts of algorithm-specific data structures into MB.
 */
class MemoryTracker {
public:
    static double bytes_to_mb(std::size_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }

    static double current_process_rss_mb() {
#if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
        }
        return 0.0;
#elif defined(__linux__)
        long rss_pages = 0;
        FILE* fp = std::fopen("/proc/self/statm", "r");
        if (fp != nullptr) {
            long total_pages = 0;
            if (std::fscanf(fp, "%ld %ld", &total_pages, &rss_pages) == 2) {
                std::fclose(fp);
                long page_size = sysconf(_SC_PAGESIZE);
                return static_cast<double>(rss_pages * page_size) / (1024.0 * 1024.0);
            }
            std::fclose(fp);
        }
        return 0.0;
#elif defined(__APPLE__)
        struct rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
        }
        return 0.0;
#else
        return 0.0;
#endif
    }
};

} // namespace incremental_search
