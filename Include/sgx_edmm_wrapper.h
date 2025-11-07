#ifndef SGX_EDMM_WRAPPER_H
#define SGX_EDMM_WRAPPER_H

#ifdef USE_SGX
#include "Enclave.h"
#include "sgx_trts.h"

// Try to include EDMM-related headers from SGX SDK 2.19+
// The header name may vary depending on SDK version
#if defined(__has_include)
  #if __has_include("sgx_rsrv_mem_mngr.h")
    #include "sgx_rsrv_mem_mngr.h"
    #define HAS_EDMM_API 1
  #elif __has_include("sgx_mm.h")
    #include "sgx_mm.h"
    #define HAS_EDMM_API 1
  #else
    #define HAS_EDMM_API 0
  #endif
#else
  // Fallback for older compilers
  #define HAS_EDMM_API 0
#endif

#else
// Non-SGX build
#define HAS_EDMM_API 0
#endif

#include <stddef.h>
#include <stdint.h>
#include <mutex>

// EDMM Statistics Structure
struct EdmmStats {
    uint64_t total_alloc_calls;
    uint64_t total_commit_calls;
    uint64_t total_decommit_calls;
    uint64_t total_bytes_reserved;
    uint64_t total_bytes_committed;
    uint64_t current_bytes_committed;
    uint64_t peak_bytes_committed;
    
    EdmmStats() : 
        total_alloc_calls(0),
        total_commit_calls(0),
        total_decommit_calls(0),
        total_bytes_reserved(0),
        total_bytes_committed(0),
        current_bytes_committed(0),
        peak_bytes_committed(0) {}
};

// EDMM Manager Class
class EdmmManager {
public:
    static EdmmManager& getInstance() {
        static EdmmManager instance;
        return instance;
    }
    
    EdmmManager(EdmmManager const&) = delete;
    void operator=(EdmmManager const&) = delete;
    
    // Reserve memory region (sgx_alloc_rsrv_mem)
    void* reserve_memory(size_t size);
    
    // Commit pages (sgx_commit_rsrv_mem)
    bool commit_pages(void* addr, size_t size);
    
    // Decommit pages (sgx_decommit_rsrv_mem)
    bool decommit_pages(void* addr, size_t size);
    
    // Free reserved memory
    void free_reserved_memory(void* addr);
    
    // Get statistics
    EdmmStats get_stats() const {
        std::unique_lock<std::mutex> lock(stats_mutex);
        return stats;
    }
    
    // Print statistics
    void print_stats() const;
    
    // Check if EDMM is available
    static bool is_edmm_available() {
#if HAS_EDMM_API
        return true;
#else
        return false;
#endif
    }

private:
    EdmmManager() {}
    
    mutable std::mutex stats_mutex;
    EdmmStats stats;
    
    void update_commit_stats(size_t size) {
        std::unique_lock<std::mutex> lock(stats_mutex);
        stats.total_commit_calls++;
        stats.total_bytes_committed += size;
        stats.current_bytes_committed += size;
        if (stats.current_bytes_committed > stats.peak_bytes_committed) {
            stats.peak_bytes_committed = stats.current_bytes_committed;
        }
    }
    
    void update_decommit_stats(size_t size) {
        std::unique_lock<std::mutex> lock(stats_mutex);
        stats.total_decommit_calls++;
        if (stats.current_bytes_committed >= size) {
            stats.current_bytes_committed -= size;
        }
    }
};

// Inline implementations
inline void* EdmmManager::reserve_memory(size_t size) {
#if HAS_EDMM_API && defined(USE_SGX)
    void* addr = nullptr;
    
    // Align size to page boundary (4KB)
    size_t page_size = 4096;
    size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
    
    #ifdef sgx_alloc_rsrv_mem
    int ret = sgx_alloc_rsrv_mem(aligned_size, &addr);
    if (ret == 0 && addr != nullptr) {
        std::unique_lock<std::mutex> lock(stats_mutex);
        stats.total_alloc_calls++;
        stats.total_bytes_reserved += aligned_size;
        return addr;
    }
    #endif
    
    return nullptr;
#else
    // Fallback: use regular malloc
    (void)size;
    return nullptr;
#endif
}

inline bool EdmmManager::commit_pages(void* addr, size_t size) {
#if HAS_EDMM_API && defined(USE_SGX)
    if (addr == nullptr || size == 0) {
        return false;
    }
    
    // Align size to page boundary (4KB)
    size_t page_size = 4096;
    size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
    
    #ifdef sgx_commit_rsrv_mem
    int ret = sgx_commit_rsrv_mem(addr, aligned_size);
    if (ret == 0) {
        update_commit_stats(aligned_size);
        return true;
    }
    #endif
    
    return false;
#else
    (void)addr;
    (void)size;
    return true; // Pretend success for non-EDMM builds
#endif
}

inline bool EdmmManager::decommit_pages(void* addr, size_t size) {
#if HAS_EDMM_API && defined(USE_SGX)
    if (addr == nullptr || size == 0) {
        return false;
    }
    
    // Align size to page boundary (4KB)
    size_t page_size = 4096;
    size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
    
    #ifdef sgx_decommit_rsrv_mem
    int ret = sgx_decommit_rsrv_mem(addr, aligned_size);
    if (ret == 0) {
        update_decommit_stats(aligned_size);
        return true;
    }
    #endif
    
    return false;
#else
    (void)addr;
    (void)size;
    return true; // Pretend success for non-EDMM builds
#endif
}

inline void EdmmManager::free_reserved_memory(void* addr) {
#if HAS_EDMM_API && defined(USE_SGX)
    if (addr == nullptr) {
        return;
    }
    
    #ifdef sgx_free_rsrv_mem
    sgx_free_rsrv_mem(addr, 0);
    #endif
#else
    (void)addr;
#endif
}

inline void EdmmManager::print_stats() const {
    EdmmStats s = get_stats();
    
#ifdef USE_SGX
    // Use OCALL to print EDMM stats
    extern sgx_status_t ocall_print_edmm_stats(
        uint64_t total_alloc,
        uint64_t total_commit,
        uint64_t total_decommit,
        uint64_t bytes_reserved,
        uint64_t bytes_committed,
        uint64_t current_committed,
        uint64_t peak_committed
    );
    
    ocall_print_edmm_stats(
        s.total_alloc_calls,
        s.total_commit_calls,
        s.total_decommit_calls,
        s.total_bytes_reserved,
        s.total_bytes_committed,
        s.current_bytes_committed,
        s.peak_bytes_committed
    );
#endif
}

#endif // SGX_EDMM_WRAPPER_H

