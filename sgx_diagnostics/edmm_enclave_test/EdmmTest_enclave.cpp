/*
 * EdmmTest_enclave.cpp — Trusted enclave code for EDMM runtime verification
 *
 * Four tests:
 *   1. Compile-time EDMM API availability check
 *   2. sgx_alloc_rsrv_mem() — reserved memory allocation (triggers EAUG on SGX2)
 *   3. sgx_mm_alloc() with COMMIT_ON_DEMAND — newer EDMM API (requires SGX2)
 *   4. Detailed EDMM runtime diagnostics (EDMM_supported flag, reserved mem info)
 */

#include "EdmmTest_t.h"
#include <sgx_trts.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

/* ── Compile-time feature detection ──────────────────────────── */

#ifdef __has_include
  #if __has_include(<sgx_rsrv_mem_mngr.h>)
    #define HAS_RSRV_MEM_API 1
    #include <sgx_rsrv_mem_mngr.h>
  #else
    #define HAS_RSRV_MEM_API 0
  #endif

  #if __has_include(<sgx_mm.h>)
    #define HAS_MM_API 1
    #include <sgx_mm.h>
  #else
    #define HAS_MM_API 0
  #endif
#else
  #define HAS_RSRV_MEM_API 0
  #define HAS_MM_API 0
#endif

/* ── Access the EDMM_supported runtime flag from libsgx_trts ─── */
/* This is a global variable defined in libsgx_trts (init_enclave.o)
 * and set during enclave initialization based on the cpuid_table
 * populated by the untrusted runtime (urts). */
extern "C" {
    extern int EDMM_supported;
}

/* ── Test 1: API availability ────────────────────────────────── */

int ecall_check_edmm_api(void)
{
    /*
     * Return a bitmask:
     *   bit 0 = sgx_rsrv_mem_mngr.h available
     *   bit 1 = sgx_mm.h available
     */
    int result = 0;
#if HAS_RSRV_MEM_API
    result |= 1;
#endif
#if HAS_MM_API
    result |= 2;
#endif
    return result;
}

/* ── Test 2: Reserved memory allocation ──────────────────────── */

int ecall_test_rsrv_alloc(uint64_t *allocated_addr, uint64_t *allocated_size)
{
    *allocated_addr = 0;
    *allocated_size = 0;

#if HAS_RSRV_MEM_API
    /* Try to allocate 4KB of reserved memory.
     * On SGX2 hardware with EDMM, this triggers EAUG to dynamically
     * add pages to the enclave. On SGX1, this only works within
     * statically pre-allocated reserved regions (if any). */
    size_t alloc_size = 4096;
    void *ptr = sgx_alloc_rsrv_mem(alloc_size);
    if (ptr != NULL) {
        *allocated_addr = (uint64_t)ptr;
        *allocated_size = alloc_size;

        /* Clean up */
        sgx_free_rsrv_mem(ptr, alloc_size);
        return 0;  /* Success */
    }
    return -2;  /* Allocation failed — likely no EDMM or no reserved region */
#else
    return -1;  /* API not available at compile time */
#endif
}

/* ── Test 3: sgx_mm_alloc with COMMIT_ON_DEMAND ─────────────── */

int ecall_test_mm_alloc(uint64_t *allocated_addr, uint64_t *allocated_size)
{
    *allocated_addr = 0;
    *allocated_size = 0;

#if HAS_MM_API
    /* sgx_mm_alloc() with SGX_EMA_COMMIT_ON_DEMAND requires true SGX2/EDMM.
     * On SGX1 hardware, this will fail because EAUG/EACCEPT are not available.
     * This is the most reliable runtime test for EDMM. */
    size_t alloc_size = 4096;
    void *addr = NULL;

    #ifdef SGX_EMA_COMMIT_ON_DEMAND
    /* sgx_mm_alloc signature:
     *   int sgx_mm_alloc(void* addr, size_t length, int flags,
     *                    sgx_enclave_fault_handler_t handler,
     *                    void* handler_private, void** out_addr)
     *
     * flags combines allocation type + page type in a single int.
     * Protection is implicit (RW for REG pages on COMMIT_ON_DEMAND). */
    int flags = SGX_EMA_COMMIT_ON_DEMAND | SGX_EMA_PAGE_TYPE_REG;
    int ret = sgx_mm_alloc(NULL, alloc_size, flags,
                           NULL,   /* fault handler */
                           NULL,   /* handler private data */
                           &addr);
    if (ret == 0 && addr != NULL) {
        *allocated_addr = (uint64_t)addr;
        *allocated_size = alloc_size;

        /* Touch the page to trigger EACCEPT */
        volatile char *p = (volatile char *)addr;
        *p = 0x42;

        /* Clean up */
        sgx_mm_dealloc(addr, alloc_size);
        return 0;  /* Success — true EDMM confirmed */
    }
    /* Return the errno from sgx_mm_alloc as a negative value offset by -100
     * so the caller can distinguish it from our error codes.
     * e.g., ret=EFAULT(14) -> return -114 */
    return -100 - ret;
    #else
    return -3;  /* SGX_EMA_COMMIT_ON_DEMAND not defined — SDK too old */
    #endif

#else
    return -1;  /* sgx_mm.h not available at compile time */
#endif
}

/* ── Test 4: Detailed EDMM runtime diagnostics ───────────────── */

int ecall_get_edmm_diagnostics(uint64_t *edmm_flag,
                               uint64_t *rsrv_base,
                               uint64_t *rsrv_max_size,
                               uint64_t *rsrv_info_ret)
{
    *edmm_flag = 0;
    *rsrv_base = 0;
    *rsrv_max_size = 0;
    *rsrv_info_ret = 0;

    /* 1. Read the EDMM_supported flag from libsgx_trts runtime.
     *    This is set by init_enclave() -> feature_supported(cpuid_table, 0).
     *    If 0, the trts believes EDMM is NOT available. */
    *edmm_flag = (uint64_t)EDMM_supported;

#if HAS_RSRV_MEM_API
    /* 2. Query reserved memory region info.
     *    sgx_get_rsrv_mem_info() returns the base and max size of the
     *    reserved memory region configured in the enclave metadata.
     *    If no ReservedMemMaxSize is in Enclave.config.xml AND EDMM is
     *    not available, both will be 0 and sgx_alloc_rsrv_mem will fail. */
    void *start_addr = NULL;
    size_t max_size = 0;
    sgx_status_t ret = sgx_get_rsrv_mem_info(&start_addr, &max_size);
    *rsrv_info_ret = (uint64_t)ret;
    *rsrv_base = (uint64_t)start_addr;
    *rsrv_max_size = (uint64_t)max_size;
#else
    *rsrv_info_ret = 0xFFFF;  /* API not available */
#endif

    return 0;
}
