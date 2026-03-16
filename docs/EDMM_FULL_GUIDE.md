# TAOISM 项目 SGX2 EDMM 启用全流程文档

> 从诊断到修复到实际 DNN 推理——在 Intel SGX Enclave 中启用动态内存管理的完整记录

## 目录

1. [背景与动机](#1-背景与动机)
2. [环境信息](#2-环境信息)
3. [诊断过程](#3-诊断过程)
4. [根因分析](#4-根因分析)
5. [修复方案](#5-修复方案)
6. [验证结果](#6-验证结果)
7. [EDMM DNN 推理示例](#7-edmm-dnn-推理示例)
8. [EDMM 对 TAOISM 的影响](#8-edmm-对-taoism-的影响)
9. [附录](#9-附录)

---

## 1. 背景与动机

### 1.1 什么是 EDMM

EDMM (Enclave Dynamic Memory Management) 是 Intel SGX2 引入的关键特性，允许在 Enclave 运行期间动态增减 EPC (Enclave Page Cache) 页面。核心指令：

| 指令 | 作用 | 执行者 |
|------|------|--------|
| `EAUG` | 向 Enclave 动态添加新页 | 内核 (Ring-0) |
| `EACCEPT` | Enclave 确认接受新页 | Enclave (Ring-3) |
| `EMODT` | 修改页类型 (REG ↔ TCS ↔ TRIM) | 内核 |
| `EMODPR` | 修改页权限 (RWX) | 内核 |

### 1.2 为什么 TAOISM 需要 EDMM

TAOISM 在 SGX Enclave 中执行 DNN 推理，典型场景需要数百 MB 到数 GB 的张量内存。传统 SGX1 模式下：

- 启动时即提交全部 `HeapMaxSize` 的 EPC 页（如 2GB）
- 即使模型只需 200MB，也占用 2GB EPC
- 多 Enclave 并行时 EPC 资源争抢严重

EDMM 模式下：

- 启动时仅提交 `HeapMinSize`（如 64KB）
- 每层推理分配张量时，由内核通过 EAUG 按需注入 EPC 页
- 推理结束后可通过 TRIM 释放不需要的页
- EPC 利用率大幅提升

---

## 2. 环境信息

| 项目 | 值 |
|------|-----|
| 机器 | Alibaba Cloud ECS (KVM 虚拟化) |
| CPU | Intel Xeon Platinum 8369B (Ice Lake-SP) |
| EPC | 3968 MB (~3.9 GB)，单 Section |
| 内核 | 6.8.0-79-generic (≥ 6.0，支持 EDMM ioctl) |
| SGX PSW | 2.28.100.1 (libsgx-urts, libsgx-enclave-common) |
| SGX SDK | 2.26.100.0 (/opt/intel/sgxsdk) |
| CPUID | SGX1 ✓, SGX2 ✓, FLC ✓, EXINFO ✓ |

---

## 3. 诊断过程

### 3.1 诊断工具套件

项目 `sgx_diagnostics/` 提供三阶段诊断：

```
sgx_diagnostics/
├── cpuid_sgx_probe.c          # Phase 1: CPUID 硬件检测
├── system_sgx_info.sh          # Phase 2: 系统级 SGX 信息
├── edmm_enclave_test/          # Phase 3: Enclave 内 EDMM 运行时测试
│   ├── EdmmTest.config.xml
│   ├── EdmmTest_enclave.cpp    # 4 个 EDMM 测试 ecall
│   ├── EdmmTest_app.cpp        # 诊断驱动 + 根因分析
│   └── Makefile
└── run_diagnostics.sh          # 总控脚本
```

### 3.2 运行诊断

```bash
cd sgx_diagnostics
source /opt/intel/sgxsdk/environment
make all
./run_diagnostics.sh
```

### 3.3 原始诊断结果（修复前）

```
Phase 1 - CPUID:
  SGX2 (EAX bit 1):   YES   ← 硬件支持
  EXINFO:              YES

Phase 3 - Enclave Runtime:
  EDMM_supported (trts):  1        ← trts 检测到 EDMM
  sgx_mm_alloc(COD):      FAILED   ← errno=12 (ENOMEM)

VERDICT: EDMM runtime tests failed.
```

**关键矛盾**：硬件支持、内核支持、trts 检测通过，但 `sgx_mm_alloc()` 返回 ENOMEM。

---

## 4. 根因分析

### 4.1 分析思路

ENOMEM 表示 EMM (Enclave Memory Manager) 找不到可用的虚拟地址空间。逐层排查：

```
CPUID SGX2=YES  →  内核 EDMM ioctl OK  →  urts 设置 cpuid_table  →  trts EDMM_supported=1
                                                                          ↓
                                                               sgx_mm_alloc(COMMIT_ON_DEMAND)
                                                                          ↓
                                                               EMM 搜索 User Region → 空！
                                                                          ↓
                                                                     ENOMEM ✗
```

### 4.2 根本原因

查阅 SDK 样例 (`/opt/intel/sgxsdk/SampleCode/SampleEnclave/Enclave/config.05.xml`)，发现：

> `sgx_mm_alloc()` 从 Enclave 的 **User Region** 中分配地址空间。User Region 必须在 `Enclave.config.xml` 中通过 `<UserRegionSize>` 显式声明。

原始配置中 **完全没有 `<UserRegionSize>`**，导致 `sgx_sign` 工具不在 ELRANGE 中划出 User Region，EMM 无地址可分配。

### 4.3 次要问题

| 问题 | 说明 |
|------|------|
| `<MiscMask>0xFFFFFFFE</MiscMask>` | bit 0=0 不强制 EXINFO，EACCEPT 可能异常 |
| 缺少 `<HeapMinSize>` | 默认等于 HeapMaxSize，堆全量提交，无 EDMM 按需增长 |
| 链接脚本缺符号 | `g_peak_heap_used` / `g_peak_rsrv_mem_committed` 未导出 |

---

## 5. 修复方案

### 5.1 Enclave.config.xml

修复前后对比：

```xml
<!-- ========== 修复前 ========== -->
<HeapMaxSize>0x80000000</HeapMaxSize>     <!-- 2GB，全量提交 -->
<!-- 无 HeapMinSize -->
<!-- 无 HeapInitSize -->
<!-- 无 UserRegionSize -->
<MiscSelect>1</MiscSelect>
<MiscMask>0xFFFFFFFE</MiscMask>           <!-- bit0=0 -->


<!-- ========== 修复后 ========== -->
<HeapMaxSize>0x80000000</HeapMaxSize>     <!-- 2GB 上限 -->
<HeapInitSize>0x80000000</HeapInitSize>   <!-- 非EDMM回退值 -->
<HeapMinSize>0x10000</HeapMinSize>        <!-- EDMM: 64KB起步 -->
<UserRegionSize>0x100000</UserRegionSize> <!-- 1MB User Region -->
<MiscSelect>1</MiscSelect>
<MiscMask>0xFFFFFFFF</MiscMask>           <!-- bit0=1，强制EXINFO -->
```

### 5.2 各参数含义

```
┌──────────────────────────────────────────────────────────┐
│                    Enclave ELRANGE (SECS.SIZE)           │
│                                                          │
│  ┌──────────┐  ┌──────────────────────────────────────┐  │
│  │ Code+Data│  │            Heap Region                │  │
│  │ (固定)   │  │                                      │  │
│  │          │  │  ┌─────────┐   ┌──────────────────┐  │  │
│  │          │  │  │HeapMin  │   │ 按需 EAUG 扩展   │  │  │
│  │          │  │  │(64KB)   │   │ → HeapMax (2GB)  │  │  │
│  │          │  │  │已提交   │   │ 未提交           │  │  │
│  │          │  │  └─────────┘   └──────────────────┘  │  │
│  │          │  │                                      │  │
│  └──────────┘  └──────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────┐  ┌──────┐  ┌─────────────────────┐  │
│  │  User Region   │  │Stack │  │  Guard / TCS / SSA  │  │
│  │  (1MB)         │  │      │  │                     │  │
│  │  sgx_mm_alloc  │  │      │  │                     │  │
│  └────────────────┘  └──────┘  └─────────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

- **HeapMinSize**: EDMM 系统上堆的初始提交量，差值部分通过 EAUG 按需增长
- **HeapInitSize**: 非 EDMM (SGX1) 系统上堆的提交量，作为回退
- **HeapMaxSize**: 堆的虚拟地址上限
- **UserRegionSize**: `sgx_mm_alloc()` 专用的独立地址区域
- **MiscMask bit 0**: 强制启用 EXINFO，EACCEPT 指令依赖此特性

### 5.3 链接脚本 (.lds)

```diff
 enclave.so
 {
     global:
         g_global_data_sim;
         g_global_data;
         enclave_entry;
+        g_peak_heap_used;
+        g_peak_rsrv_mem_committed;
     local:
         *;
 };
```

### 5.4 重新编译

```bash
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
make clean && make SGX_MODE=HW all
```

**注意**: 添加 `<UserRegionSize>` 后 `sgx_sign` 会输出：

```
INFO: Enclave configuration 'UserRegionSize' requires the enclave to be run on SGX2 platform.
generate_compatible_metadata: Only requires SGX2 metadata
```

这意味着该 Enclave **只能在 SGX2 平台上运行**，不兼容 SGX1。

---

## 6. 验证结果

### 6.1 诊断测试通过

```
========================================================
    EDMM Enclave Runtime Test (with diagnostics)
========================================================

  Enclave created successfully (EID: 2)
  MISC_SELECT:  0x00000001
  EXINFO bit:   SET (EDMM-capable enclave)

  EDMM_supported flag (trts):  1  (EDMM detected by trts)

  sgx_mm_alloc(COMMIT_ON_DEMAND):
    Result:    SUCCESS (code: 0)
    Allocated: 0x7F39D75BC000 (4096 bytes)
    -> COMMIT_ON_DEMAND succeeded. True EDMM confirmed!

  VERDICT: EDMM is fully functional.
```

### 6.2 完整诊断套件输出

```
╔══════════════════════════════════════════════════════╗
║                  FINAL VERDICT                       ║
╚══════════════════════════════════════════════════════╝

  CPUID SGX2 bit:                      YES
  Kernel EDMM support:                 YES
  Runtime: sgx_mm_alloc(COD):          SUCCESS

  ✓ EDMM is FULLY OPERATIONAL.
```

---

## 7. EDMM DNN 推理示例

### 7.1 示例说明

`experiments/models/edmm_inference_demo.py` 展示了在启用 EDMM 的 Enclave 中执行 ResNet-18 前端推理。

### 7.2 运行方式

```bash
# 1. 确保已编译 (使用修复后的 Enclave.config.xml)
make clean && make SGX_MODE=HW all

# 2. 运行示例
python -m experiments.models.edmm_inference_demo
```

### 7.3 示例输出结构

```
======================================================================
  TAOISM EDMM DNN 推理示例
  ResNet-18 Stem + Layer1.Block0 — SGX2 EDMM 动态内存按需分配
======================================================================

[1/4] 读取 Enclave 配置 (Enclave/Enclave.config.xml)
  HeapMinSize    =        0x10000  (64.0 KB，EDMM 初始提交)
  HeapMaxSize    =     0x80000000  (2.0 GB，按需增长上限)
  UserRegionSize =       0x100000  (1.0 MB，sgx_mm_alloc 专用)

[2/4] 初始化 SGX Enclave
  Enclave 初始化完成: XX ms
  EDMM 模式下堆从 64.0 KB 起步，随张量分配自动增长

[3/4] 逐层推理并观测内存增长
  层名             类型       输入形状           输出形状              分配量       累计    推理(ms)
  conv1            Conv2d     1x3x224x224        1x64x112x112       ...MB     ...MB      XX.XX
  bn1              BatchNorm  1x64x112x112       1x64x112x112       ...MB     ...MB      XX.XX
  relu1            ReLU       1x64x112x112       1x64x112x112       ...MB     ...MB      XX.XX
  ...

[4/4] EDMM 推理汇总
  SGX1: 启动即提交 2.0 GB，无论是否使用
  SGX2 EDMM: 从 64.0 KB 起步，仅提交实际使用的页
```

### 7.4 工作原理

```
时间轴:
  t0: Enclave 创建
      └─ 堆: 64KB (HeapMinSize)，其余为虚拟地址保留

  t1: conv1 权重加载 (约 9.4KB)
      └─ malloc() 触发 EAUG → 内核注入新 EPC 页 → EACCEPT
      └─ 堆增长至 ~100KB

  t2: conv1 forward (input=1x3x224x224, output=1x64x112x112)
      └─ 张量分配: 602KB (input) + 3.1MB (output) = ~3.7MB
      └─ 触发多次 EAUG，堆增长至 ~4MB

  t3: bn1 forward
      └─ 张量 + gamma/beta/running stats: ~3.2MB
      └─ 堆增长至 ~7MB

  t4: layer1_conv1 forward
      └─ 张量: ~3.2MB
      └─ 堆增长至 ~10MB

  t_end: 推理完成
      └─ 实际使用: ~10MB
      └─ SGX1 下本应提交: 2GB
      └─ EDMM 节省: ~1.99GB 的 EPC 页
```

---

## 8. EDMM 对 TAOISM 的影响

### 8.1 ChunkPool 层面

`SGXDNN/chunk_manager.cpp` 中的 `ChunkPool` 已集成 EDMM 支持：

```cpp
// 构造时尝试使用 EDMM 预留地址空间
if (EdmmManager::is_edmm_available()) {
    reserved_base = edmm_mgr.reserve_memory(total_size);
    if (reserved_base != nullptr) {
        use_edmm = true;
    }
}

// 获取 chunk 时按需提交 EPC 页
int ChunkPool::get_chunk_id() {
    if (use_edmm && !committed[res]) {
        edmm_mgr.commit_pages(chunks[res], num_byte_chunk);
        committed[res] = true;
    }
    return res;
}
```

这实现了 **Chunk 级别的按需提交**：只有在实际使用某个 Chunk 时才占用 EPC 物理页。

### 8.2 堆级别

通过 `HeapMinSize` < `HeapMaxSize` 配置，标准堆分配 (`malloc`/`new`) 也能受益于 EDMM：

- 堆扩展由 SGX SDK 的 EMM 自动管理
- 无需修改应用代码
- 对所有使用 `malloc` 的层操作透明生效

### 8.3 UserRegion 级别

通过 `<UserRegionSize>` 和 `sgx_mm_alloc()` API，可以实现更精细的内存控制：

- 手动管理 EPC 页的提交/回收
- 适合大块临时缓冲区的场景
- 可配合 `sgx_mm_dealloc()` 实现主动释放

### 8.4 性能影响

| 操作 | 开销 |
|------|------|
| 首次分配 (EAUG+EACCEPT) | ~2-5 μs/页 (一次性) |
| 后续访问 | 零额外开销 |
| 页面回收 (TRIM) | ~1-3 μs/页 |

对于 DNN 推理，EAUG 开销在模型加载时一次性摊销，推理本身不受影响。

---

## 9. 附录

### 9.1 相关文件

| 文件 | 说明 |
|------|------|
| `Enclave/Enclave.config.xml` | Enclave 配置 (EDMM 参数) |
| `Enclave/Enclave.edl` | Ecall/Ocall 接口定义 |
| `Include/sgx_edmm_wrapper.h` | EDMM Manager 封装 (统计/分配/提交/回收) |
| `SGXDNN/chunk_manager.cpp` | ChunkPool EDMM 集成 |
| `sgx_diagnostics/` | SGX2/EDMM 诊断套件 |
| `experiments/models/edmm_inference_demo.py` | EDMM 推理示例 |
| `docs/EDMM_FIX_RECORD.md` | 修复记录精要 |

### 9.2 SDK 参考配置

来自 `/opt/intel/sgxsdk/SampleCode/SampleEnclave/`：

| 配置文件 | 说明 |
|---------|------|
| `config.01.xml` | 无动态扩展 |
| `config.02.xml` | 堆动态扩展 (HeapMin < HeapMax) |
| `config.03.xml` | 动态线程 |
| `config.04.xml` | 动态线程 + 栈扩展 |
| `config.05.xml` | **UserRegionSize — sgx_mm_alloc() 专用** |

### 9.3 常见问题

**Q: 添加 UserRegionSize 后 Enclave 能否在 SGX1 机器上运行？**

不能。`sgx_sign` 输出 `Only requires SGX2 metadata`，Enclave 加载时会检查 SGX2 支持。如需兼容 SGX1，不要使用 `UserRegionSize`，改用 `HeapMinSize < HeapMaxSize` 实现堆级 EDMM。

**Q: HeapMinSize 设多大合适？**

建议设为 `0x10000` (64KB)。太小可能导致初始化时频繁 EAUG，太大则失去 EDMM 的按需分配优势。

**Q: UserRegionSize 设多大合适？**

取决于是否直接调用 `sgx_mm_alloc()`。如仅依赖堆分配，`0x100000` (1MB) 足够。如果 ChunkPool 使用 EDMM，需要根据 `STORE_CHUNK_ELEM * sizeof(float) * pool_size` 计算。

**Q: `sgx_alloc_rsrv_mem()` 和 `sgx_mm_alloc()` 有什么区别？**

- `sgx_alloc_rsrv_mem()`: 旧版 API，操作 Reserved Memory 区域（需 `ReservedMemMaxSize` 配置）
- `sgx_mm_alloc()`: 新版 API (SDK ≥ 2.22)，操作 User Region（需 `UserRegionSize` 配置）
- 推荐使用 `sgx_mm_alloc()`
