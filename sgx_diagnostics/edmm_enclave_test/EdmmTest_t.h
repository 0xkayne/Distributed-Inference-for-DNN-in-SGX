#ifndef EDMMTEST_T_H__
#define EDMMTEST_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

int ecall_check_edmm_api(void);
int ecall_test_rsrv_alloc(uint64_t* allocated_addr, uint64_t* allocated_size);
int ecall_test_mm_alloc(uint64_t* allocated_addr, uint64_t* allocated_size);
int ecall_get_edmm_diagnostics(uint64_t* edmm_flag, uint64_t* rsrv_base, uint64_t* rsrv_max_size, uint64_t* rsrv_info_ret);

sgx_status_t SGX_CDECL sgx_oc_cpuidex(int cpuinfo[4], int leaf, int subleaf);
sgx_status_t SGX_CDECL sgx_thread_wait_untrusted_event_ocall(int* retval, const void* self);
sgx_status_t SGX_CDECL sgx_thread_set_untrusted_event_ocall(int* retval, const void* waiter);
sgx_status_t SGX_CDECL sgx_thread_setwait_untrusted_events_ocall(int* retval, const void* waiter, const void* self);
sgx_status_t SGX_CDECL sgx_thread_set_multiple_untrusted_events_ocall(int* retval, const void** waiters, size_t total);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
