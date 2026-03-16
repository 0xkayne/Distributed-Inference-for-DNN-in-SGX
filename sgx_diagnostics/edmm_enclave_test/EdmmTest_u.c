#include "EdmmTest_u.h"
#include <errno.h>

typedef struct ms_ecall_check_edmm_api_t {
	int ms_retval;
} ms_ecall_check_edmm_api_t;

typedef struct ms_ecall_test_rsrv_alloc_t {
	int ms_retval;
	uint64_t* ms_allocated_addr;
	uint64_t* ms_allocated_size;
} ms_ecall_test_rsrv_alloc_t;

typedef struct ms_ecall_test_mm_alloc_t {
	int ms_retval;
	uint64_t* ms_allocated_addr;
	uint64_t* ms_allocated_size;
} ms_ecall_test_mm_alloc_t;

typedef struct ms_ecall_get_edmm_diagnostics_t {
	int ms_retval;
	uint64_t* ms_edmm_flag;
	uint64_t* ms_rsrv_base;
	uint64_t* ms_rsrv_max_size;
	uint64_t* ms_rsrv_info_ret;
} ms_ecall_get_edmm_diagnostics_t;

typedef struct ms_sgx_oc_cpuidex_t {
	int* ms_cpuinfo;
	int ms_leaf;
	int ms_subleaf;
} ms_sgx_oc_cpuidex_t;

typedef struct ms_sgx_thread_wait_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_self;
} ms_sgx_thread_wait_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_set_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_waiter;
} ms_sgx_thread_set_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_setwait_untrusted_events_ocall_t {
	int ms_retval;
	const void* ms_waiter;
	const void* ms_self;
} ms_sgx_thread_setwait_untrusted_events_ocall_t;

typedef struct ms_sgx_thread_set_multiple_untrusted_events_ocall_t {
	int ms_retval;
	const void** ms_waiters;
	size_t ms_total;
} ms_sgx_thread_set_multiple_untrusted_events_ocall_t;

static sgx_status_t SGX_CDECL EdmmTest_sgx_oc_cpuidex(void* pms)
{
	ms_sgx_oc_cpuidex_t* ms = SGX_CAST(ms_sgx_oc_cpuidex_t*, pms);
	sgx_oc_cpuidex(ms->ms_cpuinfo, ms->ms_leaf, ms->ms_subleaf);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL EdmmTest_sgx_thread_wait_untrusted_event_ocall(void* pms)
{
	ms_sgx_thread_wait_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_wait_untrusted_event_ocall_t*, pms);
	ms->ms_retval = sgx_thread_wait_untrusted_event_ocall(ms->ms_self);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL EdmmTest_sgx_thread_set_untrusted_event_ocall(void* pms)
{
	ms_sgx_thread_set_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_untrusted_event_ocall_t*, pms);
	ms->ms_retval = sgx_thread_set_untrusted_event_ocall(ms->ms_waiter);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL EdmmTest_sgx_thread_setwait_untrusted_events_ocall(void* pms)
{
	ms_sgx_thread_setwait_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_setwait_untrusted_events_ocall_t*, pms);
	ms->ms_retval = sgx_thread_setwait_untrusted_events_ocall(ms->ms_waiter, ms->ms_self);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL EdmmTest_sgx_thread_set_multiple_untrusted_events_ocall(void* pms)
{
	ms_sgx_thread_set_multiple_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_multiple_untrusted_events_ocall_t*, pms);
	ms->ms_retval = sgx_thread_set_multiple_untrusted_events_ocall(ms->ms_waiters, ms->ms_total);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[5];
} ocall_table_EdmmTest = {
	5,
	{
		(void*)EdmmTest_sgx_oc_cpuidex,
		(void*)EdmmTest_sgx_thread_wait_untrusted_event_ocall,
		(void*)EdmmTest_sgx_thread_set_untrusted_event_ocall,
		(void*)EdmmTest_sgx_thread_setwait_untrusted_events_ocall,
		(void*)EdmmTest_sgx_thread_set_multiple_untrusted_events_ocall,
	}
};
sgx_status_t ecall_check_edmm_api(sgx_enclave_id_t eid, int* retval)
{
	sgx_status_t status;
	ms_ecall_check_edmm_api_t ms;
	status = sgx_ecall(eid, 0, &ocall_table_EdmmTest, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_test_rsrv_alloc(sgx_enclave_id_t eid, int* retval, uint64_t* allocated_addr, uint64_t* allocated_size)
{
	sgx_status_t status;
	ms_ecall_test_rsrv_alloc_t ms;
	ms.ms_allocated_addr = allocated_addr;
	ms.ms_allocated_size = allocated_size;
	status = sgx_ecall(eid, 1, &ocall_table_EdmmTest, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_test_mm_alloc(sgx_enclave_id_t eid, int* retval, uint64_t* allocated_addr, uint64_t* allocated_size)
{
	sgx_status_t status;
	ms_ecall_test_mm_alloc_t ms;
	ms.ms_allocated_addr = allocated_addr;
	ms.ms_allocated_size = allocated_size;
	status = sgx_ecall(eid, 2, &ocall_table_EdmmTest, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_get_edmm_diagnostics(sgx_enclave_id_t eid, int* retval, uint64_t* edmm_flag, uint64_t* rsrv_base, uint64_t* rsrv_max_size, uint64_t* rsrv_info_ret)
{
	sgx_status_t status;
	ms_ecall_get_edmm_diagnostics_t ms;
	ms.ms_edmm_flag = edmm_flag;
	ms.ms_rsrv_base = rsrv_base;
	ms.ms_rsrv_max_size = rsrv_max_size;
	ms.ms_rsrv_info_ret = rsrv_info_ret;
	status = sgx_ecall(eid, 3, &ocall_table_EdmmTest, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

