#!/usr/bin/env bash
#
# system_sgx_info.sh — System-level SGX information aggregation
#
# Collects: CPU model, SGX flags, kernel version, device nodes,
#           EPC info, PSW/SDK versions, EDMM API headers,
#           virtualization environment, AESM service status.
#

set -euo pipefail

# ── Helpers ───────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "  ${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "  ${GREEN}[  OK]${NC}  $*"; }
warn()  { echo -e "  ${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "  ${RED}[FAIL]${NC}  $*"; }

section() {
    echo ""
    echo -e "${BOLD}────────────────────────────────────────────────────${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BOLD}────────────────────────────────────────────────────${NC}"
}

# ── 1. CPU Information ────────────────────────────────────────────

section "CPU Information"

cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
cpu_family=$(grep -m1 'cpu family' /proc/cpuinfo | cut -d: -f2 | xargs)
cpu_model_num=$(grep -m1 -w 'model' /proc/cpuinfo | cut -d: -f2 | xargs)
cpu_stepping=$(grep -m1 'stepping' /proc/cpuinfo | cut -d: -f2 | xargs)

info "Model:    $cpu_model"
info "Family:   $cpu_family  Model: $cpu_model_num  Stepping: $cpu_stepping"

# ── 2. SGX CPU Flags ─────────────────────────────────────────────

section "SGX CPU Flags"

flags=$(grep -m1 'flags' /proc/cpuinfo | cut -d: -f2)

has_sgx=0
has_sgx_lc=0

if echo "$flags" | grep -qw 'sgx'; then
    has_sgx=1
    ok "sgx         — SGX supported"
else
    fail "sgx         — NOT found"
fi

if echo "$flags" | grep -qw 'sgx_lc'; then
    has_sgx_lc=1
    ok "sgx_lc      — Flexible Launch Control supported"
else
    warn "sgx_lc      — NOT found"
fi

echo ""
info "NOTE: 'sgx_lc' (FLC) is NOT the same as SGX2."
info "      SGX2 is detected via CPUID leaf 0x12, subleaf 0, EAX bit 1."
info "      Use cpuid_sgx_probe for authoritative SGX2 detection."

# ── 3. Kernel & EDMM Support ─────────────────────────────────────

section "Kernel & EDMM Support"

kernel_ver=$(uname -r)
info "Kernel version: $kernel_ver"

# Extract major.minor
kernel_major=$(echo "$kernel_ver" | cut -d. -f1)
kernel_minor=$(echo "$kernel_ver" | cut -d. -f2)

if [ "$kernel_major" -gt 6 ] || { [ "$kernel_major" -eq 6 ] && [ "$kernel_minor" -ge 0 ]; }; then
    ok "Kernel >= 6.0 — kernel-side EDMM support present"
    kernel_edmm=1
else
    warn "Kernel < 6.0 — no kernel-side EDMM support"
    kernel_edmm=0
fi

# ── 4. SGX Device Nodes ──────────────────────────────────────────

section "SGX Device Nodes"

if [ -c /dev/sgx_enclave ]; then
    ok "/dev/sgx_enclave  — present (DCAP / in-kernel driver)"
    ls -la /dev/sgx_enclave 2>/dev/null | sed 's/^/        /'
else
    fail "/dev/sgx_enclave  — NOT found"
fi

if [ -c /dev/sgx_provision ]; then
    ok "/dev/sgx_provision — present"
else
    warn "/dev/sgx_provision — NOT found"
fi

if [ -c /dev/sgx/enclave ]; then
    info "/dev/sgx/enclave   — present (alternative path)"
fi

if [ -c /dev/isgx ]; then
    warn "/dev/isgx          — legacy out-of-tree driver detected"
fi

# ── 5. EPC Information ────────────────────────────────────────────

section "EPC Information (from dmesg)"

epc_lines=$(dmesg 2>/dev/null | grep -i 'sgx.*epc' || true)
if [ -n "$epc_lines" ]; then
    echo "$epc_lines" | while IFS= read -r line; do
        info "$line"
    done
else
    warn "No EPC information found in dmesg"
    info "Try: sudo dmesg | grep -i sgx"
fi

# Firmware bug warnings
fw_bugs=$(dmesg 2>/dev/null | grep -i 'firmware bug.*epc' || true)
if [ -n "$fw_bugs" ]; then
    echo ""
    warn "Firmware warnings detected:"
    echo "$fw_bugs" | while IFS= read -r line; do
        warn "  $line"
    done
fi

# ── 6. SGX PSW Version ───────────────────────────────────────────

section "SGX PSW (Platform Software)"

psw_version=""
if command -v dpkg-query &>/dev/null; then
    for pkg in libsgx-enclave-common sgx-aesm-service libsgx-urts libsgx-launch; do
        ver=$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null || echo "not installed")
        printf "  ${CYAN}[INFO]${NC}  %-28s %s\n" "$pkg:" "$ver"
        if [ "$pkg" = "libsgx-enclave-common" ] && [ "$ver" != "not installed" ]; then
            psw_version="$ver"
        fi
    done
elif command -v rpm &>/dev/null; then
    for pkg in libsgx-enclave-common sgx-aesm-service libsgx-urts libsgx-launch; do
        ver=$(rpm -q "$pkg" 2>/dev/null || echo "not installed")
        info "$pkg: $ver"
    done
fi

# ── 7. SGX SDK ────────────────────────────────────────────────────

section "SGX SDK"

if [ -d /opt/intel/sgxsdk ]; then
    ok "SGX SDK found at /opt/intel/sgxsdk"

    # Version from pkgconfig or release notes
    if [ -f /opt/intel/sgxsdk/pkgconfig/libsgx_urts.pc ]; then
        sdk_ver=$(grep 'Version:' /opt/intel/sgxsdk/pkgconfig/libsgx_urts.pc | awk '{print $2}')
        info "SDK version (from pkgconfig): $sdk_ver"
    fi
else
    warn "SGX SDK not found at /opt/intel/sgxsdk"
fi

# ── 8. EDMM API Headers ──────────────────────────────────────────

section "EDMM API Headers"

sdk_inc="/opt/intel/sgxsdk/include"
edmm_api_found=0

if [ -f "$sdk_inc/sgx_mm.h" ]; then
    ok "sgx_mm.h             — found (new EDMM memory management API)"
    edmm_api_found=1
else
    warn "sgx_mm.h             — NOT found"
fi

if [ -f "$sdk_inc/sgx_rsrv_mem_mngr.h" ]; then
    ok "sgx_rsrv_mem_mngr.h  — found (reserved memory API)"
    edmm_api_found=1
else
    warn "sgx_rsrv_mem_mngr.h  — NOT found"
fi

if [ -f "$sdk_inc/sgx_edger8r.h" ]; then
    info "sgx_edger8r.h        — found (standard)"
fi

if [ $edmm_api_found -eq 1 ]; then
    ok "At least one EDMM API header is available for enclave development"
else
    warn "No EDMM-specific headers found. SDK may be too old for EDMM."
fi

# ── 9. Virtualization Detection ───────────────────────────────────

section "Virtualization Environment"

if command -v systemd-detect-virt &>/dev/null; then
    virt_type=$(systemd-detect-virt 2>/dev/null || echo "none")
    info "systemd-detect-virt: $virt_type"
else
    info "systemd-detect-virt not available"
fi

# DMI-based detection
if [ -f /sys/class/dmi/id/sys_vendor ]; then
    vendor=$(cat /sys/class/dmi/id/sys_vendor 2>/dev/null || echo "unknown")
    info "System vendor: $vendor"
fi

if [ -f /sys/class/dmi/id/product_name ]; then
    product=$(cat /sys/class/dmi/id/product_name 2>/dev/null || echo "unknown")
    info "Product name:  $product"
fi

# ── 10. AESM Service ─────────────────────────────────────────────

section "AESM Service Status"

if command -v systemctl &>/dev/null; then
    aesm_status=$(systemctl is-active aesmd 2>/dev/null || echo "unknown")
    if [ "$aesm_status" = "active" ]; then
        ok "aesmd service is running"
    elif [ "$aesm_status" = "inactive" ]; then
        warn "aesmd service is inactive"
    else
        warn "aesmd service status: $aesm_status"
    fi
else
    if pgrep -x aesm_service &>/dev/null; then
        ok "aesm_service process is running"
    else
        warn "aesm_service process not found"
    fi
fi

# Check AESM socket
if [ -S /var/run/aesmd/aesm.socket ]; then
    ok "AESM socket exists: /var/run/aesmd/aesm.socket"
elif [ -S /var/opt/aesmd/data/aesm.socket ]; then
    ok "AESM socket exists: /var/opt/aesmd/data/aesm.socket"
else
    warn "AESM socket not found"
fi

# ── Summary ───────────────────────────────────────────────────────

section "Summary"

echo ""
printf "  %-35s %s\n" "CPU:" "$cpu_model"
printf "  %-35s %s\n" "SGX flag:" "$([ $has_sgx -eq 1 ] && echo 'YES' || echo 'NO')"
printf "  %-35s %s\n" "SGX_LC (FLC) flag:" "$([ $has_sgx_lc -eq 1 ] && echo 'YES' || echo 'NO')"
printf "  %-35s %s (>= 6.0: %s)\n" "Kernel:" "$kernel_ver" "$([ $kernel_edmm -eq 1 ] && echo 'YES' || echo 'NO')"
printf "  %-35s %s\n" "SGX device nodes:" "$([ -c /dev/sgx_enclave ] && echo '/dev/sgx_enclave' || echo 'MISSING')"
printf "  %-35s %s\n" "EDMM API headers:" "$([ $edmm_api_found -eq 1 ] && echo 'Available' || echo 'MISSING')"
printf "  %-35s %s\n" "AESM:" "$(systemctl is-active aesmd 2>/dev/null || echo 'unknown')"
echo ""
info "Run cpuid_sgx_probe for authoritative SGX2/EDMM CPUID detection."
echo ""
