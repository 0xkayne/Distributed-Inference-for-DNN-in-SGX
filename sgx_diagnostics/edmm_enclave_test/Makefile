######## EDMM Enclave Test — Minimal SGX Enclave for EDMM Verification ########

SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_ARCH ?= x64

ifeq ($(SGX_ARCH), x86)
	SGX_COMMON_CFLAGS := -m32
	SGX_LIBRARY_PATH := $(SGX_SDK)/lib
	SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x86/sgx_sign
	SGX_EDGER8R := $(SGX_SDK)/bin/x86/sgx_edger8r
else
	SGX_COMMON_CFLAGS := -m64
	SGX_LIBRARY_PATH := $(SGX_SDK)/lib64
	SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x64/sgx_sign
	SGX_EDGER8R := $(SGX_SDK)/bin/x64/sgx_edger8r
endif

SGX_COMMON_CFLAGS += -O2 -Wall

######## Library Names ########

ifneq ($(SGX_MODE), HW)
	Urts_Library_Name := sgx_urts_sim
	Trts_Library_Name := sgx_trts_sim
	Service_Library_Name := sgx_tservice_sim
	UAE_Service_Name := sgx_uae_service_sim
else
	Urts_Library_Name := sgx_urts
	Trts_Library_Name := sgx_trts
	Service_Library_Name := sgx_tservice
	UAE_Service_Name := sgx_uae_service
endif
Crypto_Library_Name := sgx_tcrypto

######## App Settings ########

App_Cpp_Files := EdmmTest_app.cpp
App_Include_Paths := -I. -I$(SGX_SDK)/include
App_Cpp_Flags := $(SGX_COMMON_CFLAGS) -fPIC $(App_Include_Paths) -std=c++11 -DSGX_DEBUG_FLAG=1
App_Link_Flags := $(SGX_COMMON_CFLAGS) -L$(SGX_LIBRARY_PATH) -l$(Urts_Library_Name) -l$(UAE_Service_Name) -pthread

App_Name := edmm_test_app

######## Enclave Settings ########

Enclave_Cpp_Files := EdmmTest_enclave.cpp
Enclave_Include_Paths := -I. -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx

Enclave_C_Flags := $(SGX_COMMON_CFLAGS) -nostdinc -fvisibility=hidden -fpie \
	-ffunction-sections -fdata-sections -fstack-protector-strong
Enclave_C_Flags += $(Enclave_Include_Paths)
Enclave_Cpp_Flags := $(Enclave_C_Flags) -std=c++11 -nostdinc++

Enclave_Link_Flags := $(SGX_COMMON_CFLAGS) -Wl,--no-undefined -nostdlib -nodefaultlibs \
	-nostartfiles -L$(SGX_LIBRARY_PATH) \
	-Wl,--whole-archive -l$(Trts_Library_Name) -Wl,--no-whole-archive \
	-Wl,--start-group -lsgx_tstdc -lsgx_tcxx -l$(Crypto_Library_Name) \
	-l$(Service_Library_Name) -Wl,--end-group \
	-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic \
	-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections \
	-Wl,--version-script=EdmmTest.lds

Enclave_Name := edmm_test_enclave.so
Signed_Enclave_Name := edmm_test_enclave.signed.so
Enclave_Config := EdmmTest.config.xml
Enclave_Key := EdmmTest_private.pem

######## Targets ########

.PHONY: all clean

all: $(App_Name) $(Signed_Enclave_Name)
	@echo ""
	@echo "EDMM enclave test built successfully."
	@echo "Run: ./$(App_Name)"

######## Generate signing key if needed ########

$(Enclave_Key):
	openssl genrsa -out $@ -3 3072
	@echo "GENKEY => $@"

######## Edger8r ########

EdmmTest_u.c EdmmTest_u.h: $(SGX_EDGER8R) EdmmTest.edl
	$(SGX_EDGER8R) --untrusted EdmmTest.edl --search-path $(SGX_SDK)/include
	@echo "GEN => EdmmTest_u.c EdmmTest_u.h"

EdmmTest_t.c EdmmTest_t.h: $(SGX_EDGER8R) EdmmTest.edl
	$(SGX_EDGER8R) --trusted EdmmTest.edl --search-path $(SGX_SDK)/include
	@echo "GEN => EdmmTest_t.c EdmmTest_t.h"

######## App Objects ########

EdmmTest_u.o: EdmmTest_u.c
	$(CC) $(App_Cpp_Flags) -c $< -o $@
	@echo "CC  <= $<"

EdmmTest_app.o: EdmmTest_app.cpp EdmmTest_u.h
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

$(App_Name): EdmmTest_u.o EdmmTest_app.o
	$(CXX) $^ -o $@ $(App_Link_Flags)
	@echo "LINK => $@"

######## Enclave Objects ########

EdmmTest_t.o: EdmmTest_t.c
	$(CC) $(Enclave_C_Flags) -c $< -o $@
	@echo "CC  <= $<"

EdmmTest_enclave.o: EdmmTest_enclave.cpp EdmmTest_t.h
	$(CXX) $(Enclave_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

$(Enclave_Name): EdmmTest_t.o EdmmTest_enclave.o
	$(CXX) $^ -o $@ $(Enclave_Link_Flags)
	@echo "LINK => $@"

$(Signed_Enclave_Name): $(Enclave_Name) $(Enclave_Key)
	$(SGX_ENCLAVE_SIGNER) sign -key $(Enclave_Key) -enclave $(Enclave_Name) \
		-out $@ -config $(Enclave_Config)
	@echo "SIGN => $@"

######## Clean ########

clean:
	rm -f *.o $(App_Name) $(Enclave_Name) $(Signed_Enclave_Name)
	rm -f EdmmTest_u.c EdmmTest_u.h EdmmTest_t.c EdmmTest_t.h
	rm -f $(Enclave_Key)
