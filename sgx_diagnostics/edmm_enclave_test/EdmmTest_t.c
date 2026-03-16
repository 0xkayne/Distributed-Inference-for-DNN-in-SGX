#include "EdmmTest_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define ADD_ASSIGN_OVERFLOW(a, b) (	\
	((a) += (b)) < (b)	\
)


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

static sgx_status_t SGX_CDECL sgx_ecall_check_edmm_api(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_check_edmm_api_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_check_edmm_api_t* ms = SGX_CAST(ms_ecall_check_edmm_api_t*, pms);
	ms_ecall_check_edmm_api_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_check_edmm_api_t), ms, sizeof(ms_ecall_check_edmm_api_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int _in_retval;


	_in_retval = ecall_check_edmm_api();
	if (memcpy_verw_s(&ms->ms_retval, sizeof(ms->ms_retval), &_in_retval, sizeof(_in_retval))) {
		status = SGX_ERROR_UNEXPECTED;
		goto err;
	}

err:
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_test_rsrv_alloc(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_test_rsrv_alloc_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_test_rsrv_alloc_t* ms = SGX_CAST(ms_ecall_test_rsrv_alloc_t*, pms);
	ms_ecall_test_rsrv_alloc_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_test_rsrv_alloc_t), ms, sizeof(ms_ecall_test_rsrv_alloc_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	uint64_t* _tmp_allocated_addr = __in_ms.ms_allocated_addr;
	size_t _len_allocated_addr = sizeof(uint64_t);
	uint64_t* _in_allocated_addr = NULL;
	uint64_t* _tmp_allocated_size = __in_ms.ms_allocated_size;
	size_t _len_allocated_size = sizeof(uint64_t);
	uint64_t* _in_allocated_size = NULL;
	int _in_retval;

	CHECK_UNIQUE_POINTER(_tmp_allocated_addr, _len_allocated_addr);
	CHECK_UNIQUE_POINTER(_tmp_allocated_size, _len_allocated_size);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_allocated_addr != NULL && _len_allocated_addr != 0) {
		if ( _len_allocated_addr % sizeof(*_tmp_allocated_addr) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_allocated_addr = (uint64_t*)malloc(_len_allocated_addr)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_allocated_addr, 0, _len_allocated_addr);
	}
	if (_tmp_allocated_size != NULL && _len_allocated_size != 0) {
		if ( _len_allocated_size % sizeof(*_tmp_allocated_size) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_allocated_size = (uint64_t*)malloc(_len_allocated_size)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_allocated_size, 0, _len_allocated_size);
	}
	_in_retval = ecall_test_rsrv_alloc(_in_allocated_addr, _in_allocated_size);
	if (memcpy_verw_s(&ms->ms_retval, sizeof(ms->ms_retval), &_in_retval, sizeof(_in_retval))) {
		status = SGX_ERROR_UNEXPECTED;
		goto err;
	}
	if (_in_allocated_addr) {
		if (memcpy_verw_s(_tmp_allocated_addr, _len_allocated_addr, _in_allocated_addr, _len_allocated_addr)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_allocated_size) {
		if (memcpy_verw_s(_tmp_allocated_size, _len_allocated_size, _in_allocated_size, _len_allocated_size)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_allocated_addr) free(_in_allocated_addr);
	if (_in_allocated_size) free(_in_allocated_size);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_test_mm_alloc(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_test_mm_alloc_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_test_mm_alloc_t* ms = SGX_CAST(ms_ecall_test_mm_alloc_t*, pms);
	ms_ecall_test_mm_alloc_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_test_mm_alloc_t), ms, sizeof(ms_ecall_test_mm_alloc_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	uint64_t* _tmp_allocated_addr = __in_ms.ms_allocated_addr;
	size_t _len_allocated_addr = sizeof(uint64_t);
	uint64_t* _in_allocated_addr = NULL;
	uint64_t* _tmp_allocated_size = __in_ms.ms_allocated_size;
	size_t _len_allocated_size = sizeof(uint64_t);
	uint64_t* _in_allocated_size = NULL;
	int _in_retval;

	CHECK_UNIQUE_POINTER(_tmp_allocated_addr, _len_allocated_addr);
	CHECK_UNIQUE_POINTER(_tmp_allocated_size, _len_allocated_size);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_allocated_addr != NULL && _len_allocated_addr != 0) {
		if ( _len_allocated_addr % sizeof(*_tmp_allocated_addr) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_allocated_addr = (uint64_t*)malloc(_len_allocated_addr)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_allocated_addr, 0, _len_allocated_addr);
	}
	if (_tmp_allocated_size != NULL && _len_allocated_size != 0) {
		if ( _len_allocated_size % sizeof(*_tmp_allocated_size) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_allocated_size = (uint64_t*)malloc(_len_allocated_size)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_allocated_size, 0, _len_allocated_size);
	}
	_in_retval = ecall_test_mm_alloc(_in_allocated_addr, _in_allocated_size);
	if (memcpy_verw_s(&ms->ms_retval, sizeof(ms->ms_retval), &_in_retval, sizeof(_in_retval))) {
		status = SGX_ERROR_UNEXPECTED;
		goto err;
	}
	if (_in_allocated_addr) {
		if (memcpy_verw_s(_tmp_allocated_addr, _len_allocated_addr, _in_allocated_addr, _len_allocated_addr)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_allocated_size) {
		if (memcpy_verw_s(_tmp_allocated_size, _len_allocated_size, _in_allocated_size, _len_allocated_size)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_allocated_addr) free(_in_allocated_addr);
	if (_in_allocated_size) free(_in_allocated_size);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_get_edmm_diagnostics(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_get_edmm_diagnostics_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_get_edmm_diagnostics_t* ms = SGX_CAST(ms_ecall_get_edmm_diagnostics_t*, pms);
	ms_ecall_get_edmm_diagnostics_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_get_edmm_diagnostics_t), ms, sizeof(ms_ecall_get_edmm_diagnostics_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	uint64_t* _tmp_edmm_flag = __in_ms.ms_edmm_flag;
	size_t _len_edmm_flag = sizeof(uint64_t);
	uint64_t* _in_edmm_flag = NULL;
	uint64_t* _tmp_rsrv_base = __in_ms.ms_rsrv_base;
	size_t _len_rsrv_base = sizeof(uint64_t);
	uint64_t* _in_rsrv_base = NULL;
	uint64_t* _tmp_rsrv_max_size = __in_ms.ms_rsrv_max_size;
	size_t _len_rsrv_max_size = sizeof(uint64_t);
	uint64_t* _in_rsrv_max_size = NULL;
	uint64_t* _tmp_rsrv_info_ret = __in_ms.ms_rsrv_info_ret;
	size_t _len_rsrv_info_ret = sizeof(uint64_t);
	uint64_t* _in_rsrv_info_ret = NULL;
	int _in_retval;

	CHECK_UNIQUE_POINTER(_tmp_edmm_flag, _len_edmm_flag);
	CHECK_UNIQUE_POINTER(_tmp_rsrv_base, _len_rsrv_base);
	CHECK_UNIQUE_POINTER(_tmp_rsrv_max_size, _len_rsrv_max_size);
	CHECK_UNIQUE_POINTER(_tmp_rsrv_info_ret, _len_rsrv_info_ret);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_edmm_flag != NULL && _len_edmm_flag != 0) {
		if ( _len_edmm_flag % sizeof(*_tmp_edmm_flag) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_edmm_flag = (uint64_t*)malloc(_len_edmm_flag)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_edmm_flag, 0, _len_edmm_flag);
	}
	if (_tmp_rsrv_base != NULL && _len_rsrv_base != 0) {
		if ( _len_rsrv_base % sizeof(*_tmp_rsrv_base) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_rsrv_base = (uint64_t*)malloc(_len_rsrv_base)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_rsrv_base, 0, _len_rsrv_base);
	}
	if (_tmp_rsrv_max_size != NULL && _len_rsrv_max_size != 0) {
		if ( _len_rsrv_max_size % sizeof(*_tmp_rsrv_max_size) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_rsrv_max_size = (uint64_t*)malloc(_len_rsrv_max_size)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_rsrv_max_size, 0, _len_rsrv_max_size);
	}
	if (_tmp_rsrv_info_ret != NULL && _len_rsrv_info_ret != 0) {
		if ( _len_rsrv_info_ret % sizeof(*_tmp_rsrv_info_ret) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_rsrv_info_ret = (uint64_t*)malloc(_len_rsrv_info_ret)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_rsrv_info_ret, 0, _len_rsrv_info_ret);
	}
	_in_retval = ecall_get_edmm_diagnostics(_in_edmm_flag, _in_rsrv_base, _in_rsrv_max_size, _in_rsrv_info_ret);
	if (memcpy_verw_s(&ms->ms_retval, sizeof(ms->ms_retval), &_in_retval, sizeof(_in_retval))) {
		status = SGX_ERROR_UNEXPECTED;
		goto err;
	}
	if (_in_edmm_flag) {
		if (memcpy_verw_s(_tmp_edmm_flag, _len_edmm_flag, _in_edmm_flag, _len_edmm_flag)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_rsrv_base) {
		if (memcpy_verw_s(_tmp_rsrv_base, _len_rsrv_base, _in_rsrv_base, _len_rsrv_base)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_rsrv_max_size) {
		if (memcpy_verw_s(_tmp_rsrv_max_size, _len_rsrv_max_size, _in_rsrv_max_size, _len_rsrv_max_size)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_rsrv_info_ret) {
		if (memcpy_verw_s(_tmp_rsrv_info_ret, _len_rsrv_info_ret, _in_rsrv_info_ret, _len_rsrv_info_ret)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_edmm_flag) free(_in_edmm_flag);
	if (_in_rsrv_base) free(_in_rsrv_base);
	if (_in_rsrv_max_size) free(_in_rsrv_max_size);
	if (_in_rsrv_info_ret) free(_in_rsrv_info_ret);
	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[4];
} g_ecall_table = {
	4,
	{
		{(void*)(uintptr_t)sgx_ecall_check_edmm_api, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_test_rsrv_alloc, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_test_mm_alloc, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_get_edmm_diagnostics, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[5][4];
} g_dyn_entry_table = {
	5,
	{
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
	}
};


sgx_status_t SGX_CDECL sgx_oc_cpuidex(int cpuinfo[4], int leaf, int subleaf)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_cpuinfo = 4 * sizeof(int);

	ms_sgx_oc_cpuidex_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_oc_cpuidex_t);
	void *__tmp = NULL;

	void *__tmp_cpuinfo = NULL;

	CHECK_ENCLAVE_POINTER(cpuinfo, _len_cpuinfo);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (cpuinfo != NULL) ? _len_cpuinfo : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_oc_cpuidex_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_oc_cpuidex_t));
	ocalloc_size -= sizeof(ms_sgx_oc_cpuidex_t);

	if (cpuinfo != NULL) {
		if (memcpy_verw_s(&ms->ms_cpuinfo, sizeof(int*), &__tmp, sizeof(int*))) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp_cpuinfo = __tmp;
		if (_len_cpuinfo % sizeof(*cpuinfo) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		memset_verw(__tmp_cpuinfo, 0, _len_cpuinfo);
		__tmp = (void *)((size_t)__tmp + _len_cpuinfo);
		ocalloc_size -= _len_cpuinfo;
	} else {
		ms->ms_cpuinfo = NULL;
	}

	if (memcpy_verw_s(&ms->ms_leaf, sizeof(ms->ms_leaf), &leaf, sizeof(leaf))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	if (memcpy_verw_s(&ms->ms_subleaf, sizeof(ms->ms_subleaf), &subleaf, sizeof(subleaf))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(0, ms);

	if (status == SGX_SUCCESS) {
		if (cpuinfo) {
			if (memcpy_s((void*)cpuinfo, _len_cpuinfo, __tmp_cpuinfo, _len_cpuinfo)) {
				sgx_ocfree();
				return SGX_ERROR_UNEXPECTED;
			}
		}
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_wait_untrusted_event_ocall(int* retval, const void* self)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_wait_untrusted_event_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_wait_untrusted_event_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t);

	if (memcpy_verw_s(&ms->ms_self, sizeof(ms->ms_self), &self, sizeof(self))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(1, ms);

	if (status == SGX_SUCCESS) {
		if (retval) {
			if (memcpy_s((void*)retval, sizeof(*retval), &ms->ms_retval, sizeof(ms->ms_retval))) {
				sgx_ocfree();
				return SGX_ERROR_UNEXPECTED;
			}
		}
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_set_untrusted_event_ocall(int* retval, const void* waiter)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_set_untrusted_event_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_set_untrusted_event_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_set_untrusted_event_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_set_untrusted_event_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_set_untrusted_event_ocall_t);

	if (memcpy_verw_s(&ms->ms_waiter, sizeof(ms->ms_waiter), &waiter, sizeof(waiter))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(2, ms);

	if (status == SGX_SUCCESS) {
		if (retval) {
			if (memcpy_s((void*)retval, sizeof(*retval), &ms->ms_retval, sizeof(ms->ms_retval))) {
				sgx_ocfree();
				return SGX_ERROR_UNEXPECTED;
			}
		}
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_setwait_untrusted_events_ocall(int* retval, const void* waiter, const void* self)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_setwait_untrusted_events_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_setwait_untrusted_events_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t);

	if (memcpy_verw_s(&ms->ms_waiter, sizeof(ms->ms_waiter), &waiter, sizeof(waiter))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	if (memcpy_verw_s(&ms->ms_self, sizeof(ms->ms_self), &self, sizeof(self))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(3, ms);

	if (status == SGX_SUCCESS) {
		if (retval) {
			if (memcpy_s((void*)retval, sizeof(*retval), &ms->ms_retval, sizeof(ms->ms_retval))) {
				sgx_ocfree();
				return SGX_ERROR_UNEXPECTED;
			}
		}
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_set_multiple_untrusted_events_ocall(int* retval, const void** waiters, size_t total)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_waiters = total * sizeof(void*);

	ms_sgx_thread_set_multiple_untrusted_events_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(waiters, _len_waiters);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (waiters != NULL) ? _len_waiters : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_set_multiple_untrusted_events_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t);

	if (waiters != NULL) {
		if (memcpy_verw_s(&ms->ms_waiters, sizeof(const void**), &__tmp, sizeof(const void**))) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		if (_len_waiters % sizeof(*waiters) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_verw_s(__tmp, ocalloc_size, waiters, _len_waiters)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_waiters);
		ocalloc_size -= _len_waiters;
	} else {
		ms->ms_waiters = NULL;
	}

	if (memcpy_verw_s(&ms->ms_total, sizeof(ms->ms_total), &total, sizeof(total))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(4, ms);

	if (status == SGX_SUCCESS) {
		if (retval) {
			if (memcpy_s((void*)retval, sizeof(*retval), &ms->ms_retval, sizeof(ms->ms_retval))) {
				sgx_ocfree();
				return SGX_ERROR_UNEXPECTED;
			}
		}
	}
	sgx_ocfree();
	return status;
}

