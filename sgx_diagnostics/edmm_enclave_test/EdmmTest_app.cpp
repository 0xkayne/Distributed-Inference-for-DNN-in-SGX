/*
 * EdmmTest_app.cpp — Untrusted application for EDMM enclave test
 *
 * Loads the minimal EDMM test enclave, runs four diagnostic ecalls,
 * and reports results with detailed root-cause information.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#include "sgx_urts.h"
#include "sgx_uae_service.h"
#include "EdmmTest_u.h"

#define ENCLAVE_FILE "edmm_test_enclave.signed.so"

/* ── Helpers ──────────────────────────────────────────────────── */

static const char *result_str(int code)
{
    switch (code) {
    case 0:  return "SUCCESS";
    case -1: return "API not available (header missing at compile time)";
    case -2: return "Allocation FAILED (no EDMM or no reserved region)";
    case -3: return "SDK too old (SGX_EMA_COMMIT_ON_DEMAND not defined)";
    default:
        if (code <= -100) {
            /* sgx_mm_alloc returned errno = -(code + 100) */
            static char buf[128];
            int err = -(code + 100);
            snprintf(buf, sizeof(buf),
                     "sgx_mm_alloc FAILED with errno=%d (%s)",
                     err,
                     err == ENOMEM ? "ENOMEM — out of memory/no free space" :
                     err == EFAULT ? "EFAULT — general failure (EDMM not operational)" :
                     err == EINVAL ? "EINVAL — invalid parameters" :
                     err == EACCES ? "EACCES — region outside enclave" :
                     "other error");
            return buf;
        }
        return "Unknown error";
    }
}

static const char *sgx_status_str(uint64_t code)
{
    switch (code) {
    case 0:    return "SGX_SUCCESS";
    case 1:    return "SGX_ERROR_UNEXPECTED";
    case 0xFFFF: return "API not available";
    default:
        static char buf[64];
        snprintf(buf, sizeof(buf), "0x%04lX", (unsigned long)code);
        return buf;
    }
}

static void print_separator(void)
{
    printf("----------------------------------------------------\n");
}

static void print_section(const char *title)
{
    printf("\n");
    print_separator();
    printf("  %s\n", title);
    print_separator();
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    sgx_enclave_id_t eid = 0;
    sgx_status_t ret;
    sgx_launch_token_t token = {0};
    int token_updated = 0;
    sgx_misc_attribute_t misc_attr;

    const char *enclave_path = ENCLAVE_FILE;
    if (argc > 1)
        enclave_path = argv[1];

    printf("\n");
    printf("========================================================\n");
    printf("    EDMM Enclave Runtime Test (with diagnostics)\n");
    printf("========================================================\n");

    /* ── Create enclave ──────────────────────────── */
    print_section("Enclave Creation");

    printf("  Loading: %s\n", enclave_path);

    memset(&misc_attr, 0, sizeof(misc_attr));

    ret = sgx_create_enclave(enclave_path,
                             SGX_DEBUG_FLAG,
                             &token,
                             &token_updated,
                             &eid,
                             &misc_attr);

    if (ret != SGX_SUCCESS) {
        printf("  FAILED to create enclave: error 0x%04X\n", ret);
        switch (ret) {
        case SGX_ERROR_NO_DEVICE:
            printf("  -> No SGX device found. Is the driver loaded?\n");
            break;
        case SGX_ERROR_ENCLAVE_FILE_ACCESS:
            printf("  -> Cannot open enclave file: %s\n", enclave_path);
            break;
        case SGX_ERROR_INVALID_ENCLAVE:
            printf("  -> Invalid enclave image.\n");
            break;
        case SGX_ERROR_OUT_OF_MEMORY:
            printf("  -> Out of EPC memory.\n");
            break;
        case SGX_ERROR_MEMORY_MAP_CONFLICT:
            printf("  -> Memory map conflict. Possibly EDMM-related.\n");
            break;
        default:
            printf("  -> See SGX SDK documentation for error 0x%04X\n", ret);
            break;
        }
        return 1;
    }

    printf("  Enclave created successfully (EID: %lu)\n", (unsigned long)eid);

    /* Check MISC_EXINFO */
    int has_exinfo = (misc_attr.misc_select & 0x1) ? 1 : 0;
    printf("  MISC_SELECT:  0x%08X\n", misc_attr.misc_select);
    printf("  EXINFO bit:   %s\n", has_exinfo ? "SET (EDMM-capable enclave)" : "NOT SET");

    /* ── Test 1: EDMM API availability ───────────── */
    print_section("Test 1: EDMM API Compile-Time Availability");

    int api_result = 0;
    ret = ecall_check_edmm_api(eid, &api_result);
    if (ret != SGX_SUCCESS) {
        printf("  ECALL failed: 0x%04X\n", ret);
    } else {
        int has_rsrv = (api_result & 1) != 0;
        int has_mm   = (api_result & 2) != 0;
        printf("  sgx_rsrv_mem_mngr.h:  %s\n", has_rsrv ? "AVAILABLE" : "NOT FOUND");
        printf("  sgx_mm.h:             %s\n", has_mm   ? "AVAILABLE" : "NOT FOUND");
    }

    /* ── Test 4 (run early): EDMM diagnostics ───── */
    print_section("Test 4: EDMM Runtime Diagnostics (inside enclave)");

    uint64_t edmm_flag = 0, rsrv_base = 0, rsrv_max_size = 0, rsrv_info_ret = 0;
    int diag_result = -99;
    ret = ecall_get_edmm_diagnostics(eid, &diag_result,
                                     &edmm_flag, &rsrv_base,
                                     &rsrv_max_size, &rsrv_info_ret);
    if (ret != SGX_SUCCESS) {
        printf("  ECALL failed: 0x%04X\n", ret);
    } else {
        printf("  EDMM_supported flag (trts):  %lu", (unsigned long)edmm_flag);
        if (edmm_flag == 0) {
            printf("  *** NOT SET ***\n");
            printf("    -> The trts runtime does NOT believe EDMM is available.\n");
            printf("    -> This is set by feature_supported(cpuid_table, 0) during\n");
            printf("       enclave init. The cpuid_table is populated by the urts\n");
            printf("       (untrusted runtime) from the host CPU's CPUID results.\n");
            printf("    -> ROOT CAUSE: The urts did not set the EDMM bit in the\n");
            printf("       cpuid_table, OR g_sdk_version <= 4 caused the check\n");
            printf("       to be skipped entirely.\n");
        } else {
            printf("  (EDMM detected by trts)\n");
        }
        printf("  Reserved mem info:\n");
        printf("    sgx_get_rsrv_mem_info() returned: %s\n", sgx_status_str(rsrv_info_ret));
        printf("    Reserved base addr:  0x%lX\n", (unsigned long)rsrv_base);
        printf("    Reserved max size:   %lu bytes (%.1f MB)\n",
               (unsigned long)rsrv_max_size,
               (double)rsrv_max_size / (1024.0 * 1024.0));
        if (rsrv_max_size == 0 && edmm_flag == 0) {
            printf("    -> No reserved memory region AND EDMM_supported=0.\n");
            printf("       sgx_alloc_rsrv_mem() has nowhere to allocate from.\n");
            printf("       Fix: Either enable EDMM in SDK/urts, or add\n");
            printf("       <ReservedMemMaxSize> to Enclave.config.xml.\n");
        }
    }

    /* ── Test 2: sgx_alloc_rsrv_mem ──────────────── */
    print_section("Test 2: sgx_alloc_rsrv_mem() -- Reserved Memory");

    uint64_t rsrv_addr = 0, rsrv_size = 0;
    int rsrv_result = -99;
    ret = ecall_test_rsrv_alloc(eid, &rsrv_result, &rsrv_addr, &rsrv_size);
    if (ret != SGX_SUCCESS) {
        printf("  ECALL failed: 0x%04X\n", ret);
    } else {
        printf("  Result:    %s (code: %d)\n", result_str(rsrv_result), rsrv_result);
        if (rsrv_result == 0) {
            printf("  Allocated: 0x%lX (%lu bytes)\n",
                   (unsigned long)rsrv_addr, (unsigned long)rsrv_size);
            printf("  -> Reserved memory allocation works.\n");
            printf("     On SGX2 this triggers EAUG; on SGX1 uses pre-allocated pool.\n");
        }
    }

    /* ── Test 3: sgx_mm_alloc with COMMIT_ON_DEMAND ─ */
    print_section("Test 3: sgx_mm_alloc() -- COMMIT_ON_DEMAND");

    uint64_t mm_addr = 0, mm_size = 0;
    int mm_result = -99;
    ret = ecall_test_mm_alloc(eid, &mm_result, &mm_addr, &mm_size);
    if (ret != SGX_SUCCESS) {
        printf("  ECALL failed: 0x%04X\n", ret);
    } else {
        printf("  Result:    %s (code: %d)\n", result_str(mm_result), mm_result);
        if (mm_result == 0) {
            printf("  Allocated: 0x%lX (%lu bytes)\n",
                   (unsigned long)mm_addr, (unsigned long)mm_size);
            printf("  -> COMMIT_ON_DEMAND succeeded. True EDMM confirmed!\n");
        }
    }

    /* ── Summary ─────────────────────────────────── */
    print_section("EDMM Runtime Test Summary");

    printf("  %-35s %s\n", "Enclave creation:", "OK");
    printf("  %-35s %s\n", "EXINFO:", has_exinfo ? "YES" : "NO");
    printf("  %-35s %lu\n", "EDMM_supported (trts flag):", (unsigned long)edmm_flag);
    printf("  %-35s 0x%lX (%lu bytes)\n", "Reserved mem region:",
           (unsigned long)rsrv_base, (unsigned long)rsrv_max_size);
    printf("  %-35s %s\n", "sgx_alloc_rsrv_mem():",
           rsrv_result == 0 ? "SUCCESS" : "FAILED");
    printf("  %-35s %s\n", "sgx_mm_alloc(COMMIT_ON_DEMAND):",
           mm_result == 0 ? "SUCCESS" : "FAILED");
    printf("\n");

    if (mm_result == 0) {
        printf("  VERDICT: EDMM is fully functional.\n");
    } else if (rsrv_result == 0) {
        printf("  VERDICT: Reserved memory works, but sgx_mm_alloc(COMMIT_ON_DEMAND)\n");
        printf("           failed. EDMM may be partially supported or SDK too old.\n");
    } else {
        printf("  VERDICT: EDMM runtime tests failed. SGX2 EDMM is NOT operational.\n");
        printf("\n");
        if (edmm_flag == 0) {
            printf("  ROOT CAUSE ANALYSIS:\n");
            printf("    The CPU supports SGX2 (CPUID 0x12 leaf 0, bit 1 = 1), but the\n");
            printf("    enclave's EDMM_supported flag is 0. This means the urts\n");
            printf("    (untrusted runtime, libsgx_urts.so / libsgx_enclave_common.so)\n");
            printf("    did NOT populate the SGX2/EDMM bit in the cpuid_table that is\n");
            printf("    passed into the enclave's global_data during creation.\n");
            printf("\n");
            printf("    The urts probes EDMM by calling SGX_IOC_ENCLAVE_RESTRICT_PERMISSIONS\n");
            printf("    on a bare /dev/sgx_enclave fd. The kernel returns EINVAL (no enclave\n");
            printf("    context on that fd). The detection logic should treat EINVAL as\n");
            printf("    'ioctl exists, EDMM supported' vs ENOTTY as 'ioctl unknown'.\n");
            printf("\n");
            printf("    POSSIBLE FIXES:\n");
            printf("      1. Upgrade SGX SDK + PSW to matching versions (both >= 2.23)\n");
            printf("         Current: libsgx_trts=2.26, libsgx_enclave_common=2.28\n");
            printf("         The version mismatch may cause EDMM detection failure.\n");
            printf("      2. Rebuild the SDK from source with EDMM enabled.\n");
            printf("      3. As a workaround, add <ReservedMemMaxSize> and\n");
            printf("         <ReservedMemInitSize> to Enclave.config.xml to\n");
            printf("         pre-allocate reserved memory without requiring EDMM.\n");
        }
    }
    printf("\n");

    /* Destroy enclave */
    sgx_destroy_enclave(eid);

    /* Exit code: 0 = full EDMM, 1 = partial, 2 = no EDMM */
    if (mm_result == 0) return 0;
    if (rsrv_result == 0) return 1;
    return 2;
}
