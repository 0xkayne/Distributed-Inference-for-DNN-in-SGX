# STORE_CHUNK_ELEM 快速参考

## Inception V3 分组配置

| 组名 | STORE_CHUNK_ELEM | 内存/Chunk | 总内存(8 chunks) | 层数 | 关键约束 |
|------|-----------------|-----------|-----------------|------|---------|
| **Stem** | 130560500 | 498.05 MB | 3.89 GB | ~12 | MaxPool(35×35, 73×73) |
| **Inception-A** | 940800 | 3.59 MB | 0.03 GB | ~45 | MaxPool(35×35), OutputC |
| **Reduction-A** | 134175475 | 511.84 MB | 4.00 GB | ~9 | MaxPool(35×35, 17×17) |
| **Inception-B** | 221952 | 0.85 MB | 0.01 GB | ~60 | MaxPool(17×17), OutputC |
| **Reduction-B** | 1109760 | 4.23 MB | 0.03 GB | ~9 | MaxPool(17×17, 8×8), OutputC |
| **Inception-C** | 30720 | 0.12 MB | 0.00 GB | ~30 | MaxPool(8×8), OutputC |
| **Classifier** | 256000 | 0.98 MB | 0.01 GB | ~4 | MaxPool(8×8), Linear(2048) |

## 快速修改命令

```bash
# Stem 组
sed -i 's/#define STORE_CHUNK_ELEM [0-9]*/#define STORE_CHUNK_ELEM 130560500/' Include/common_with_enclaves.h
sed -i 's/#define WORK_CHUNK_ELEM [0-9]*/#define WORK_CHUNK_ELEM 130560500/' Include/common_with_enclaves.h

# Inception-A 组
sed -i 's/#define STORE_CHUNK_ELEM [0-9]*/#define STORE_CHUNK_ELEM 940800/' Include/common_with_enclaves.h
sed -i 's/#define WORK_CHUNK_ELEM [0-9]*/#define WORK_CHUNK_ELEM 940800/' Include/common_with_enclaves.h

# Reduction-A 组
sed -i 's/#define STORE_CHUNK_ELEM [0-9]*/#define STORE_CHUNK_ELEM 134175475/' Include/common_with_enclaves.h
sed -i 's/#define WORK_CHUNK_ELEM [0-9]*/#define WORK_CHUNK_ELEM 134175475/' Include/common_with_enclaves.h

# Inception-B 组
sed -i 's/#define STORE_CHUNK_ELEM [0-9]*/#define STORE_CHUNK_ELEM 221952/' Include/common_with_enclaves.h
sed -i 's/#define WORK_CHUNK_ELEM [0-9]*/#define WORK_CHUNK_ELEM 221952/' Include/common_with_enclaves.h

# Reduction-B 组
sed -i 's/#define STORE_CHUNK_ELEM [0-9]*/#define STORE_CHUNK_ELEM 1109760/' Include/common_with_enclaves.h
sed -i 's/#define WORK_CHUNK_ELEM [0-9]*/#define WORK_CHUNK_ELEM 1109760/' Include/common_with_enclaves.h

# Inception-C 组
sed -i 's/#define STORE_CHUNK_ELEM [0-9]*/#define STORE_CHUNK_ELEM 30720/' Include/common_with_enclaves.h
sed -i 's/#define WORK_CHUNK_ELEM [0-9]*/#define WORK_CHUNK_ELEM 30720/' Include/common_with_enclaves.h

# Classifier 组
sed -i 's/#define STORE_CHUNK_ELEM [0-9]*/#define STORE_CHUNK_ELEM 256000/' Include/common_with_enclaves.h
sed -i 's/#define WORK_CHUNK_ELEM [0-9]*/#define WORK_CHUNK_ELEM 256000/' Include/common_with_enclaves.h
```

## 约束条件说明

### MaxPool 约束（强制性）
- `STORE_CHUNK_ELEM % (input_height * input_width) == 0`
- 如果不满足，MaxPool 层会直接返回，导致错误

### Conv 约束（警告性）
- `STORE_CHUNK_ELEM % (input_row_size * stride) == 0`
- `STORE_CHUNK_ELEM % output_channel == 0`
- 如果不满足，会打印警告但代码仍可运行

### Linear 约束（重要）
- `STORE_CHUNK_ELEM % input_features == 0`
- 如果不满足，可能导致性能问题

## 内存计算

```
每个 chunk 内存 = STORE_CHUNK_ELEM * 4 bytes (float32)
总内存 = 8 chunks * 每个 chunk 内存
```

## 注意事项

1. **Reduction-A 组** 内存需求最大（4GB），确保系统有足够 EPC 内存
2. 每次修改后必须重新编译：`make clean && make`
3. 建议按顺序执行各组，因为层之间有依赖关系





