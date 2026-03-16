#!/usr/bin/env bash
#
# run_diagnostics.sh — Master SGX2/EDMM diagnostic runner
#
# Builds and runs all diagnostic tools, then produces a combined verdict.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── State variables for final verdict ─────────────────────────────

SGX2_CPUID=0       # CPUID 0x12:0 EAX bit 1
KERNEL_EDMM=0      # kernel >= 6.0 && /dev/sgx_enclave
RUNTIME_RSRV=0     # ecall_test_rsrv_alloc succeeded
RUNTIME_MM=0       # ecall_test_mm_alloc succeeded
CPUID_RAN=0
ENCLAVE_RAN=0

# ══════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║      SGX2 / EDMM Comprehensive Diagnostics           ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"

# ── Phase 1: CPUID Probe ─────────────────────────────────────────

echo ""
echo -e "${BOLD}══ Phase 1: CPUID-Level SGX2 Detection ═════════════════${NC}"
echo ""

if [ ! -f cpuid_sgx_probe ] || [ cpuid_sgx_probe.c -nt cpuid_sgx_probe ]; then
    echo "  Building cpuid_sgx_probe..."
    if cc -O2 -Wall -o cpuid_sgx_probe cpuid_sgx_probe.c 2>&1; then
        echo -e "  ${GREEN}Build successful.${NC}"
    else
        echo -e "  ${RED}Build failed. Ensure gcc/cc is installed.${NC}"
    fi
fi

if [ -x cpuid_sgx_probe ]; then
    cpuid_exit=0
    ./cpuid_sgx_probe || cpuid_exit=$?
    CPUID_RAN=1
    if [ $cpuid_exit -eq 0 ]; then
        SGX2_CPUID=1
    fi
else
    echo -e "  ${RED}cpuid_sgx_probe not available.${NC}"
fi

# ── Phase 2: System Information ───────────────────────────────────

echo ""
echo -e "${BOLD}══ Phase 2: System-Level SGX Information ════════════════${NC}"

if [ -x system_sgx_info.sh ] || [ -f system_sgx_info.sh ]; then
    bash system_sgx_info.sh
else
    echo -e "  ${RED}system_sgx_info.sh not found.${NC}"
fi

# Determine kernel EDMM support
kernel_major=$(uname -r | cut -d. -f1)
kernel_minor=$(uname -r | cut -d. -f2)
if [ "$kernel_major" -gt 6 ] || { [ "$kernel_major" -eq 6 ] && [ "$kernel_minor" -ge 0 ]; }; then
    if [ -c /dev/sgx_enclave ]; then
        KERNEL_EDMM=1
    fi
fi

# ── Phase 3: EDMM Enclave Runtime Test ────────────────────────────

echo ""
echo -e "${BOLD}══ Phase 3: EDMM Enclave Runtime Verification ══════════${NC}"
echo ""

SGX_SDK="${SGX_SDK:-/opt/intel/sgxsdk}"

if [ ! -d "$SGX_SDK" ]; then
    echo -e "  ${YELLOW}SGX SDK not found at $SGX_SDK${NC}"
    echo "  Skipping enclave runtime test."
    echo "  Set SGX_SDK=/path/to/sgxsdk and re-run."
else
    # Source SDK environment if not already done
    if [ -f "$SGX_SDK/environment" ]; then
        source "$SGX_SDK/environment" 2>/dev/null || true
    fi

    cd edmm_enclave_test

    # Build if needed
    if [ ! -f edmm_test_app ] || [ ! -f edmm_test_enclave.signed.so ]; then
        echo "  Building EDMM enclave test..."
        if make SGX_SDK="$SGX_SDK" SGX_MODE="${SGX_MODE:-HW}" 2>&1 | sed 's/^/    /'; then
            echo -e "  ${GREEN}Build successful.${NC}"
        else
            echo -e "  ${RED}Build failed. Check SGX SDK installation.${NC}"
            cd "$SCRIPT_DIR"
        fi
    fi

    if [ -x edmm_test_app ] && [ -f edmm_test_enclave.signed.so ]; then
        echo ""
        enclave_exit=0
        ./edmm_test_app || enclave_exit=$?
        ENCLAVE_RAN=1

        case $enclave_exit in
            0) RUNTIME_RSRV=1; RUNTIME_MM=1 ;;
            1) RUNTIME_RSRV=1; RUNTIME_MM=0 ;;
            *) RUNTIME_RSRV=0; RUNTIME_MM=0 ;;
        esac
    else
        echo -e "  ${YELLOW}Enclave test binary not available. Skipping.${NC}"
    fi

    cd "$SCRIPT_DIR"
fi

# ── Final Verdict ─────────────────────────────────────────────────

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                  FINAL VERDICT                       ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

printf "  %-40s " "CPUID SGX2 bit (leaf 0x12:0 EAX[1]):"
if [ $CPUID_RAN -eq 0 ]; then
    echo -e "${YELLOW}NOT TESTED${NC}"
elif [ $SGX2_CPUID -eq 1 ]; then
    echo -e "${GREEN}YES${NC}"
else
    echo -e "${RED}NO${NC}"
fi

printf "  %-40s " "Kernel EDMM support (>= 6.0 + device):"
if [ $KERNEL_EDMM -eq 1 ]; then
    echo -e "${GREEN}YES${NC}"
else
    echo -e "${RED}NO${NC}"
fi

printf "  %-40s " "Runtime: sgx_alloc_rsrv_mem():"
if [ $ENCLAVE_RAN -eq 0 ]; then
    echo -e "${YELLOW}NOT TESTED${NC}"
elif [ $RUNTIME_RSRV -eq 1 ]; then
    echo -e "${GREEN}SUCCESS${NC}"
else
    echo -e "${RED}FAILED${NC}"
fi

printf "  %-40s " "Runtime: sgx_mm_alloc(COMMIT_ON_DEMAND):"
if [ $ENCLAVE_RAN -eq 0 ]; then
    echo -e "${YELLOW}NOT TESTED${NC}"
elif [ $RUNTIME_MM -eq 1 ]; then
    echo -e "${GREEN}SUCCESS${NC}"
else
    echo -e "${RED}FAILED${NC}"
fi

echo ""
echo -e "  ${BOLD}Diagnosis:${NC}"
echo ""

if [ $SGX2_CPUID -eq 1 ] && [ $KERNEL_EDMM -eq 1 ] && [ $RUNTIME_MM -eq 1 ]; then
    echo -e "  ${GREEN}✓ EDMM is FULLY OPERATIONAL.${NC}"
    echo "    SGX2 instructions (EAUG/EMODT/EMODPR/EACCEPT) are available."
    echo "    Dynamic memory management inside enclaves will work."
    FINAL_EXIT=0

elif [ $SGX2_CPUID -eq 1 ] && [ $KERNEL_EDMM -eq 1 ] && [ $RUNTIME_RSRV -eq 1 ]; then
    echo -e "  ${YELLOW}△ EDMM partially available.${NC}"
    echo "    sgx_alloc_rsrv_mem() works, but sgx_mm_alloc(COMMIT_ON_DEMAND) failed."
    echo "    Reserved memory allocation OK; full dynamic allocation may need SDK update."
    FINAL_EXIT=0

elif [ $SGX2_CPUID -eq 1 ] && [ $KERNEL_EDMM -eq 1 ]; then
    echo -e "  ${YELLOW}△ SGX2 hardware present, kernel ready, but runtime test failed.${NC}"
    echo "    Check enclave configuration (MiscSelect/MiscMask in config XML)."
    echo "    Ensure SGX SDK >= 2.19 with EDMM API support."
    FINAL_EXIT=1

elif [ $KERNEL_EDMM -eq 1 ] && [ $SGX2_CPUID -eq 0 ]; then
    echo -e "  ${RED}✗ CPU/hypervisor does NOT expose SGX2.${NC}"
    echo "    Kernel and drivers are ready, but CPUID leaf 0x12 EAX bit 1 = 0."
    if grep -q hypervisor /proc/cpuinfo 2>/dev/null; then
        echo "    Running in a VM — the hypervisor likely does not pass through SGX2."
        echo "    Contact your cloud provider or configure KVM to expose SGX2."
    else
        echo "    This CPU does not support SGX2 EDMM instructions."
    fi
    FINAL_EXIT=1

elif [ $SGX2_CPUID -eq 1 ] && [ $KERNEL_EDMM -eq 0 ]; then
    echo -e "  ${RED}✗ CPU supports SGX2, but kernel/driver not ready.${NC}"
    echo "    Upgrade to kernel >= 6.0 and ensure SGX driver is loaded."
    FINAL_EXIT=1

else
    echo -e "  ${RED}✗ No SGX2/EDMM support detected.${NC}"
    if [ $CPUID_RAN -eq 1 ] && [ $SGX2_CPUID -eq 0 ]; then
        echo "    CPUID confirms SGX2 is not available."
    fi
    echo "    EDMM dynamic memory management will NOT work."
    echo "    The project will fall back to static memory allocation."
    FINAL_EXIT=1
fi

echo ""
echo -e "  ${BOLD}Impact on TAOISM project:${NC}"
if [ $SGX2_CPUID -eq 1 ] && [ $KERNEL_EDMM -eq 1 ]; then
    echo "    EDMM-based chunk management (ChunkPool dynamic resize) should work."
    echo "    Build with: make SGX_MODE=HW all"
else
    echo "    EDMM is not available. Use static memory allocation."
    echo "    Ensure STORE_CHUNK_ELEM and HeapMaxSize are correctly sized."
    echo "    Build with: make SGX_MODE=HW all  (static allocation fallback)"
fi
echo ""

exit ${FINAL_EXIT:-1}
