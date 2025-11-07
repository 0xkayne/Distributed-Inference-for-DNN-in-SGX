#!/bin/bash

# SGX2 EDMM Hardware and Software Capability Check Script
# This script verifies that the system supports SGX2 with EDMM features

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "   SGX2 EDMM Capability Check"
echo "================================================"
echo ""

# Check 1: CPU SGX Support
echo -n "[1/7] Checking CPU SGX support... "
if grep -q sgx /proc/cpuinfo; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "      CPU does not support SGX. Please check BIOS settings."
    exit 1
fi

# Check 2: SGX2 Support (check for sgx_lc - Launch Control)
echo -n "[2/7] Checking SGX2 (Flexible Launch Control) support... "
if grep -q sgx_lc /proc/cpuinfo; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "      SGX2/FLC not detected in /proc/cpuinfo. Continuing anyway..."
fi

# Check 3: DCAP Driver - /dev/sgx_enclave
echo -n "[3/7] Checking DCAP driver (/dev/sgx_enclave)... "
if [ -e /dev/sgx_enclave ]; then
    echo -e "${GREEN}PASS${NC}"
elif [ -e /dev/isgx ]; then
    echo -e "${YELLOW}WARNING${NC}"
    echo "      Found legacy driver /dev/isgx. EDMM requires DCAP driver (/dev/sgx_enclave)."
    echo "      Please upgrade to DCAP driver >= 1.41"
else
    echo -e "${RED}FAIL${NC}"
    echo "      No SGX device found. Please install SGX driver."
    exit 1
fi

# Check 4: DCAP Driver - /dev/sgx_provision (optional but recommended)
echo -n "[4/7] Checking provisioning device (/dev/sgx_provision)... "
if [ -e /dev/sgx_provision ]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "      /dev/sgx_provision not found. Remote attestation may not work."
fi

# Check 5: SGX SDK Installation
echo -n "[5/7] Checking SGX SDK installation... "
if [ -z "$SGX_SDK" ]; then
    if [ -d "/opt/intel/sgxsdk" ]; then
        export SGX_SDK=/opt/intel/sgxsdk
        echo -e "${GREEN}PASS${NC} (found at /opt/intel/sgxsdk)"
    else
        echo -e "${RED}FAIL${NC}"
        echo "      SGX_SDK not set and /opt/intel/sgxsdk not found."
        echo "      Please install SGX SDK >= 2.19 and source the environment."
        exit 1
    fi
else
    echo -e "${GREEN}PASS${NC} (SGX_SDK=$SGX_SDK)"
fi

# Check 6: SGX SDK Version
echo -n "[6/7] Checking SGX SDK version... "
if [ -f "$SGX_SDK/version" ]; then
    SDK_VERSION=$(cat "$SGX_SDK/version" 2>/dev/null || echo "unknown")
    echo -e "${GREEN}$SDK_VERSION${NC}"
    # Try to parse version (expecting format like "2.19" or "2.19.100.3")
    MAJOR_VERSION=$(echo "$SDK_VERSION" | cut -d. -f1)
    MINOR_VERSION=$(echo "$SDK_VERSION" | cut -d. -f2)
    if [ "$MAJOR_VERSION" -ge 3 ] || ([ "$MAJOR_VERSION" -eq 2 ] && [ "$MINOR_VERSION" -ge 19 ]); then
        echo "      Version check: ${GREEN}PASS${NC} (>= 2.19 required)"
    else
        echo "      Version check: ${YELLOW}WARNING${NC} (>= 2.19 recommended for EDMM)"
    fi
else
    echo -e "${YELLOW}UNKNOWN${NC}"
    echo "      Cannot determine SDK version. Continuing anyway..."
fi

# Check 7: EDMM Header Files
echo -n "[7/7] Checking EDMM API headers... "
EDMM_HEADERS_FOUND=0
if [ -f "$SGX_SDK/include/sgx_rsrv_mem_mngr.h" ]; then
    EDMM_HEADERS_FOUND=1
    echo -e "${GREEN}PASS${NC}"
    echo "      Found: sgx_rsrv_mem_mngr.h"
elif [ -f "$SGX_SDK/include/sgx_mm.h" ]; then
    EDMM_HEADERS_FOUND=1
    echo -e "${GREEN}PASS${NC}"
    echo "      Found: sgx_mm.h"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "      EDMM headers not found in $SGX_SDK/include/"
    echo "      This may indicate an older SDK version."
fi

echo ""
echo "================================================"
echo "   Summary"
echo "================================================"
echo ""

if [ $EDMM_HEADERS_FOUND -eq 1 ]; then
    echo -e "${GREEN}✓${NC} System appears to support SGX2 with EDMM"
    echo ""
    echo "Next steps:"
    echo "  1. Source SGX SDK environment: source $SGX_SDK/environment"
    echo "  2. Build with: make clean && make"
    echo "  3. Run tests: teeslice/scripts/run_resnet_baseline.sh"
    exit 0
else
    echo -e "${YELLOW}⚠${NC} System has SGX but EDMM support is uncertain"
    echo ""
    echo "Recommendations:"
    echo "  - Upgrade SGX SDK to version 2.19 or higher"
    echo "  - Install DCAP driver 1.41 or higher"
    echo "  - Check CPU specifications for SGX2 support"
    exit 2
fi

