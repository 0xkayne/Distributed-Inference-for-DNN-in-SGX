# SGX2 EDMM Migration Summary

## Overview

TAOISM 框架已成功从 SGX1 迁移到 SGX2 EDMM (Enclave Dynamic Memory Management)，实现了动态内存管理以提升 EPC 利用效率。

## Migration Date

**Completed:** November 5, 2025

## What Changed

### 1. Hardware Detection Script
- **Created:** `scripts/check_sgx2_edmm.sh`
- **Purpose:** 验证系统是否支持 SGX2/EDMM
- **Features:**
  - 检测 CPU SGX2 支持 (Flexible Launch Control)
  - 验证 DCAP 驱动 (`/dev/sgx_enclave`, `/dev/sgx_provision`)
  - 检查 SGX SDK 版本和 EDMM API 头文件

### 2. Enclave Configuration
- **File:** `Enclave/Enclave.config.xml`
- **Changes:**
  - 增加 `HeapMaxSize` 到 1GB 以支持动态分配
  - 设置 `MiscSelect=1` 和 `MiscMask=0xFFFFFFFE` 启用 SGX2 动态特性
  - 更新 ISVSVN 到版本 1

### 3. EDMM API Wrapper
- **Created:** `Include/sgx_edmm_wrapper.h`
- **Features:**
  - 封装 `sgx_alloc_rsrv_mem`, `sgx_commit_rsrv_mem`, `sgx_decommit_rsrv_mem`
  - `EdmmManager` 单例类管理动态内存
  - 统计 commit/decommit 操作的次数和字节数
  - 自动检测 EDMM 可用性并回退到传统 malloc

### 4. Application Side Changes
- **File:** `App/enclave_bridge.cpp`
- **Changes:**
  - 使用 `sgx_create_enclave_ex` 创建支持 KSS 的 Enclave
  - 添加 EDMM 特性检测和验证
  - 实现 `ocall_print_edmm_stats` OCALL 打印统计信息

### 5. Enclave Side Memory Management
- **Files:** `SGXDNN/chunk_manager.hpp`, `SGXDNN/chunk_manager.cpp`
- **Changes:**
  - `ChunkPool` 使用 EDMM 预留大块内存区域
  - `get_chunk_id()` 时动态 commit EPC 页面
  - `return_chunk_id()` 保持页面 committed (lazy decommit)
  - 析构函数自动 decommit 并释放保留内存
  - 自动回退到传统 `memalign` 如果 EDMM 不可用

### 6. EDL Updates
- **File:** `Enclave/Enclave.edl`
- **Added:** `ocall_print_edmm_stats` OCALL 用于输出内存统计

### 7. Makefile Enhancements
- **File:** `Makefile`
- **Added:** `check-edmm` 目标在构建前验证环境
- **Updated:** 注释说明 SGX2 要求

### 8. Documentation
- **File:** `README.md`
- **Updates:**
  - 新增 "SGX2 EDMM Requirements" 章节
  - 详细的安装步骤（DCAP 驱动、SDK 2.19+）
  - 硬件检测指南
  - 新增 "Troubleshooting SGX2 EDMM" 章节
  - 常见问题和解决方案

## Key Features

### Dynamic Memory Management
- **Reserve:** 使用 `sgx_alloc_rsrv_mem` 预留虚拟地址空间
- **Commit on Demand:** chunk 被请求时才提交 EPC 页面
- **Lazy Decommit:** 默认保持 committed 以提升性能
- **Fallback Support:** 自动回退到传统静态分配

### Memory Efficiency
- **Before (SGX1):** 所有 chunk 预先分配，占用大量 EPC
- **After (SGX2):** 按需提交，减少 EPC 峰值占用
- **Statistics:** 实时跟踪 commit/decommit 操作

### Backward Compatibility
- 如果 EDMM API 不可用，自动使用 `memalign`
- 支持旧版 SDK 的编译时检测 (`#ifdef SGX_CREATE_ENCLAVE_EX_KSS`)
- Enclave 创建失败时提供清晰的错误信息

## Testing & Verification

### Build Verification
```bash
# 1. Check EDMM support
bash scripts/check_sgx2_edmm.sh

# 2. Build with EDMM
source /opt/intel/sgxsdk/environment
make clean && make

# 3. Verify output
ls -lh enclave.signed.so App/bin/enclave_bridge.so
```

### Runtime Verification
```bash
# Activate conda environment
conda activate taoism

# Set library path
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH

# Run tests
python teeslice/eval_sgx_teeslice.py
```

### Success Indicators
- ✓ Enclave initialization message: "Initializing Enclave with SGX2 EDMM support..."
- ✓ Feature detection: "SGX2 EDMM features detected and enabled"
- ✓ EDMM statistics printed during execution (if enabled)
- ✓ No "Out of EPC memory" errors

## Performance Considerations

### Pros
- Reduced EPC peak usage
- Better memory utilization for variable workloads
- Supports larger models within EPC constraints

### Cons
- Commit operation overhead on first access
- Slightly increased latency for cold chunks

### Optimization Tips
1. **Adjust chunk size** in `Include/common_with_enclaves.h`
2. **Enable aggressive decommit** in `chunk_manager.cpp` if memory is tight
3. **Pre-commit hot chunks** for frequently used tensors
4. **Monitor EPC usage** with `sudo perf stat -e sgx:*`

## System Requirements

### Hardware
- Intel CPU with SGX2 support (Ice Lake or newer)
- Minimum 64MB EPC (recommended: 128MB+)

### Software
- Intel SGX DCAP Driver >= 1.41
- Intel SGX SDK >= 2.19
- Linux kernel with SGX support
- Python 3.7+, PyTorch 1.7.0+

## Troubleshooting

### Common Issues
1. **"EDMM features not detected"**
   - Check CPU specifications for SGX2 support
   - Verify SDK version: `cat $SGX_SDK/version`
   - Ensure DCAP driver is installed

2. **"Out of EPC memory"**
   - Adjust `HeapMaxSize` in `Enclave.config.xml`
   - Reduce `STORE_CHUNK_ELEM` in `common_with_enclaves.h`
   - Close other SGX applications

3. **Library version conflicts**
   - Set `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH`
   - Use system libstdc++ instead of conda's version

## Migration Checklist

- [x] Create hardware detection script
- [x] Update Enclave configuration for EDMM
- [x] Implement EDMM API wrapper
- [x] Modify App initialization to use sgx_create_enclave_ex
- [x] Refactor ChunkPool for dynamic memory
- [x] Add EDMM statistics OCALL
- [x] Update Makefile
- [x] Update documentation
- [x] Compile and test

## Future Work

### Potential Enhancements
1. **Adaptive Decommit:** Intelligently decommit based on memory pressure
2. **Pre-commit Strategy:** Pre-commit frequently accessed chunks
3. **EDMM Statistics Dashboard:** Real-time visualization of EPC usage
4. **Multi-tier Memory:** Combine EDMM with paging for very large models
5. **Performance Profiling:** Detailed analysis of EDMM overhead

### Known Limitations
1. EDMM not available in simulation mode (`SGX_MODE=SIM`)
2. Page commit overhead on first access
3. Current SDK may not expose all EDMM APIs
4. Limited to single enclave per process

## References

- [Intel SGX Developer Reference](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html)
- [SGX2 and EDMM Whitepaper](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sgx-sdk-developer-reference-for-linux-os.html)
- [DCAP Driver Installation](https://github.com/intel/SGXDataCenterAttestationPrimitives)
- [TAOISM Framework Paper](https://arxiv.org/abs/2310.07152)

## Contributors

Migration implemented by AI Assistant on November 5, 2025.

## License

Same as TAOISM project: MIT License

---

**Note:** This is a research prototype. Do not use in production environments without thorough security review and testing.

