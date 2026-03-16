# SGX2 EDMM 启用修复记录

> 日期: 2026-03-16 | 平台: Alibaba Cloud ECS (Intel Xeon Platinum 8369B, Ice Lake-SP)

## 一、问题现象

Enclave 内调用 `sgx_mm_alloc(COMMIT_ON_DEMAND)` 返回 `ENOMEM (errno=12)`，EDMM 动态内存管理不可用。

CPUID 确认硬件支持 SGX2（EAUG/EACCEPT），内核 6.8 支持 EDMM ioctl，trts 运行时 `EDMM_supported=1`——三层前置条件全部满足，唯独 **Enclave 配置缺少关键字段**。

## 二、根本原因

`Enclave.config.xml` 缺少 `<UserRegionSize>` 配置。

`sgx_mm_alloc()` 在 Enclave 内部从 EMM（Enclave Memory Manager）管理的 **User Region** 中分配虚拟地址空间，再由内核通过 EAUG 指令按需注入物理 EPC 页。如果 config 中没有声明 `<UserRegionSize>`，sgx_sign 工具不会在 ELRANGE 中划出 User Region，EMM 找不到可用地址空间，直接返回 ENOMEM。

同时还有两个次要问题：

| 问题 | 原值 | 修正值 | 影响 |
|------|------|--------|------|
| `MiscMask` | `0xFFFFFFFE` | `0xFFFFFFFF` | bit 0=0 时不强制检查 EXINFO，EACCEPT 指令可能异常 |
| 缺少 `HeapMinSize` | 默认=HeapMaxSize | `0x10000` (64KB) | 堆无法按需增长，EDMM 对堆的优化失效 |
| 链接脚本缺符号 | 无 | `g_peak_heap_used` / `g_peak_rsrv_mem_committed` | SDK 运行时需要这些符号做内存统计 |

## 三、修复方案

### Enclave.config.xml 关键配置

```xml
<!-- 堆：小初始值 + 大上限 = EDMM 按需增长 -->
<HeapMaxSize>0x80000000</HeapMaxSize>       <!-- 2GB 上限 -->
<HeapInitSize>0x80000000</HeapInitSize>     <!-- 非EDMM回退：全量提交 -->
<HeapMinSize>0x10000</HeapMinSize>          <!-- EDMM模式：64KB起步，按需增长 -->

<!-- sgx_mm_alloc() 专用地址空间 -->
<UserRegionSize>0x100000</UserRegionSize>   <!-- 1MB User Region -->

<!-- EXINFO 强制启用（EACCEPT 依赖） -->
<MiscSelect>1</MiscSelect>
<MiscMask>0xFFFFFFFF</MiscMask>
```

### 链接脚本 (.lds) 导出符号

```lds
global:
    g_global_data_sim;
    g_global_data;
    enclave_entry;
    g_peak_heap_used;           /* 新增 */
    g_peak_rsrv_mem_committed;  /* 新增 */
```

## 四、验证结果

```
EDMM_supported flag (trts):         1   (trts 检测到 EDMM)
sgx_mm_alloc(COMMIT_ON_DEMAND):     SUCCESS (0x7F39D75BC000, 4096 bytes)
VERDICT: EDMM is fully functional.
```

## 五、核心要点

1. **`UserRegionSize` 是 `sgx_mm_alloc()` 的前提** —— 没有它，EMM 无地址空间可分配
2. **`HeapMinSize` 控制 EDMM 模式下堆的初始提交量** —— 差值部分按需 EAUG
3. **`HeapInitSize` 是非 EDMM 系统的回退值** —— SGX1 机器使用此值
4. **添加 `UserRegionSize` 后 Enclave 变为 SGX2-only** —— 无法在 SGX1 平台运行
5. **SDK 参考**: `/opt/intel/sgxsdk/SampleCode/SampleEnclave/Enclave/config.05.xml`
