# SGX2/EDMM Diagnostics Suite
#
# Targets:
#   make probe     — Build CPUID probe (no SDK needed)
#   make enclave   — Build EDMM enclave test (needs SGX SDK)
#   make all       — Build everything available
#   make clean     — Clean all build artifacts

SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW

.PHONY: all probe enclave clean

all: probe enclave

# ── CPUID Probe (no dependencies) ────────────────────────────────

probe: cpuid_sgx_probe

cpuid_sgx_probe: cpuid_sgx_probe.c
	$(CC) -O2 -Wall -o $@ $<
	@echo "BUILD => $@  (no SDK required)"

# ── EDMM Enclave Test (requires SGX SDK) ─────────────────────────

enclave:
	@if [ -d "$(SGX_SDK)" ]; then \
		echo "SGX SDK found, building EDMM enclave test..."; \
		$(MAKE) -C edmm_enclave_test SGX_SDK=$(SGX_SDK) SGX_MODE=$(SGX_MODE); \
	else \
		echo "SGX SDK not found at $(SGX_SDK), skipping enclave test."; \
		echo "Set SGX_SDK=/path/to/sgxsdk to build."; \
	fi

# ── Clean ────────────────────────────────────────────────────────

clean:
	rm -f cpuid_sgx_probe
	$(MAKE) -C edmm_enclave_test clean 2>/dev/null || true
