# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TAOISM is a research framework for DNN topology-based distributed parallel inference acceleration in Intel SGX Trusted Execution Environments. It implements a cost measurement framework and distributed inference engine supporting arbitrary DAG topologies for multi-partition parallel execution across CPU and SGX Enclave boundaries.

The README and most documentation is written in Chinese.

## Build Commands

```bash
# Environment setup (required before building or running)
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
conda activate taoism

# Build (hardware mode, default)
make SGX_MODE=HW all

# Build (simulation mode, no SGX hardware needed)
make SGX_MODE=SIM SGX_DEBUG=1 SGX_PRERELEASE=0 all

# Clean and rebuild
make clean && make

# After changing STORE_CHUNK_ELEM or enclave config:
rm -rf SGXDNN/bin_sgx && make clean && make

# Verify SGX2 EDMM support
make check-edmm
```

## Running Experiments

```bash
# Quick validation test
python experiments/quick_test.py

# Profiling (use -m for module invocation from project root)
python -m experiments.models.profile_resnet_enclave
python -m experiments.models.profile_bert_enclave --model mini --per-head

# Distributed inference
python -m experiments.models.distributed_resnet

# Measurement scripts
python -m experiments.measurement.measure_computation
python -m experiments.measurement.measure_communication
```

## Architecture

### Three-Layer Stack

1. **C/C++ SGX Layer** (`Enclave/`, `SGXDNN/`, `App/`): Trusted enclave code implementing DNN operations with encrypted memory. The EDL interface (`Enclave/Enclave.edl`) defines 30+ ecalls for layer operations (conv, linear, maxpool, batchnorm, layernorm, softmax, gelu, matmul) and tensor management.

2. **Python Interface** (`python/`): Bridges Python and the C enclave via `enclave_interfaces.py`. `sgx_net.py` provides the neural network construction framework. Layer implementations live in `python/layers/`.

3. **Experiments** (`experiments/`): Model definitions (`experiments/models/`), measurement scripts (`experiments/measurement/`), and data collection (`experiments/data/`).

### Chunk-Based Memory Management

Large tensors are split into fixed-size chunks controlled by `STORE_CHUNK_ELEM` in `Include/common_with_enclaves.h`. Chunks inside the EPC are plaintext; chunks evicted to untrusted memory are AES-GCM encrypted. The `ChunkPool` and `TrustedChunkManager` in `SGXDNN/chunk_manager.cpp` handle this lifecycle.

**Critical**: `STORE_CHUNK_ELEM` must be tuned per model/input size. Typical values:
- NiN/ResNet-18 (32x32 input): `409600`
- VGG16/AlexNet (224x224): `802816`
- Large models (Inception V3 299x299): much larger values needed

Enclave heap is configured in `Enclave/Enclave.config.xml` (default 2GB).

### Distributed Inference

`FlexibleGraphWorker` (in `experiments/models/distributed_resnet.py` and similar) implements thread-based distributed execution. It performs topology analysis to find "cut edges" between partitions, creates inter-partition communication queues, and runs partitions in parallel threads. A shared model instance with locks avoids GlobalTensor conflicts.

### Attention Module

`python/layers/attention/` provides a unified multi-head attention implementation supporting BERT, ALBERT, DistilBERT, TinyBERT, ViT, and Swin Transformer. Uses a factory pattern (`attention_factory.py`) with two modes: `batched` (production) and `per_head` (fine-grained profiling).

## Supported Models

**Vision**: NiN, VGG16, ResNet-18, AlexNet, Inception V3/V4, ViT, Swin Transformer, ResNeXt
**NLP**: BERT, ALBERT, DistilBERT, TinyBERT

## Key Configuration Files

- `Include/common_with_enclaves.h` — `STORE_CHUNK_ELEM`, `THREAD_POOL_SIZE`, type definitions
- `Enclave/Enclave.config.xml` — Heap size, stack size, TCS count, EDMM settings
- `Enclave/Enclave.edl` — Ecall interface definitions between trusted/untrusted code

## Requirements

- Intel CPU with SGX2 support (Ice Lake+), >=128MB EPC (BIOS configured)
- Ubuntu 20.04 LTS, Intel SGX SDK >=2.19, Python 3.7+, PyTorch >=1.7.0
- Eigen library is vendored in `Include/eigen/`
