#ifdef USE_SGX
#include "Enclave.h"
#endif

#include "matmul.hpp"
#include "chunk_manager.hpp"
#include "layer_timing.hpp"
#include "runtime_stats_api.hpp"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

using namespace std;

MatMulBuffer::MatMulBuffer(IdT FunId_) : FunId(FunId_) {}

void MatMulBuffer::init(
    IdT input1, IdT input2, IdT output,
    uint32_t batch_, uint32_t num_heads_, uint32_t seq_len_,
    uint32_t dim1_, uint32_t dim2_,
    bool transpose_a_, bool transpose_b_,
    float scale_) {
    
    input1_tensor = GetTenById(input1);
    input2_tensor = GetTenById(input2);
    output_tensor = GetTenById(output);
    
    batch = batch_;
    num_heads = num_heads_;
    seq_len = seq_len_;
    dim1 = dim1_;
    dim2 = dim2_;
    transpose_a = transpose_a_;
    transpose_b = transpose_b_;
    scale = scale_;
    
    // Compute M, K, N based on transpose flags
    // For Q @ K^T: A is (B, H, N, D), B is (B, H, N, D) with trans_b=true
    //   M = seq_len, K = dim1 (head_dim), N = seq_len
    // For Attn @ V: A is (B, H, N, N), B is (B, H, N, D)
    //   M = seq_len, K = seq_len, N = dim2 (head_dim)
    
    if (transpose_a) {
        M = dim1;
        K = seq_len;
    } else {
        M = seq_len;
        K = dim1;
    }
    
    if (transpose_b) {
        N = seq_len;
    } else {
        N = dim2;
    }
}

void MatMulBuffer::matmul_single(
    DtypeForCpuOp* A, DtypeForCpuOp* B, DtypeForCpuOp* C,
    int M_, int K_, int N_, bool trans_a, bool trans_b) {
    
    // Initialize output to zero
    memset(C, 0, M_ * N_ * sizeof(DtypeForCpuOp));
    
    // Naive matrix multiplication (can be optimized with blocking/SIMD)
    for (int i = 0; i < M_; i++) {
        for (int k = 0; k < K_; k++) {
            // Get A[i, k] considering transpose
            DtypeForCpuOp a_val;
            if (trans_a) {
                a_val = A[k * M_ + i];  // A is K x M, access as A[k][i]
            } else {
                a_val = A[i * K_ + k];  // A is M x K, access as A[i][k]
            }
            
            for (int j = 0; j < N_; j++) {
                // Get B[k, j] considering transpose
                DtypeForCpuOp b_val;
                if (trans_b) {
                    b_val = B[j * K_ + k];  // B is N x K, access as B[j][k]
                } else {
                    b_val = B[k * N_ + j];  // B is K x N, access as B[k][j]
                }
                
                C[i * N_ + j] += a_val * b_val;
            }
        }
    }
    
    // Apply scaling if needed
    if (scale != 1.0f) {
        for (int i = 0; i < M_ * N_; i++) {
            C[i] *= scale;
        }
    }
}

void MatMulBuffer::forward() {
    auto& chunk_manager = TrustedChunkManager::getInstance();

    // For batch matmul, we process each (batch, head) pair independently
    int num_batch_head = batch * num_heads;
    
    // Sizes for each matrix in a single (batch, head)
    int size_A = seq_len * dim1;   // Input1 matrix size per head
    int size_B = seq_len * dim2;   // Input2 matrix size per head (or dim1 * seq_len if transposed)
    int size_C = M * N;            // Output matrix size per head
    
    // Allocate working chunks
    DtypeForCpuOp *A_chunk, *B_chunk, *C_chunk;
    ChunkGuard<DtypeForCpuOp> A_guard(StoreChunkPool::GetChunkPool(), A_chunk);
    ChunkGuard<DtypeForCpuOp> B_guard(StoreChunkPool::GetChunkPool(), B_chunk);
    ChunkGuard<DtypeForCpuOp> C_guard(StoreChunkPool::GetChunkPool(), C_chunk);

    double get1_ms = 0.0;
    double get2_ms = 0.0;
    double compute_ms = 0.0;
    double store_ms = 0.0;

    // Helper: get number of elements in a given chunk index for a tensor
    auto chunk_elems_for = [&](shared_ptr<SecretTen> ten, int chunk_idx) -> int {
        int total = ten->GetNumElem();
        int start = chunk_idx * STORE_CHUNK_ELEM;
        int remain = total - start;
        return (remain >= STORE_CHUNK_ELEM) ? STORE_CHUNK_ELEM : remain;
    };

    // Helper: read a contiguous slice [offset, offset+len) (in elements) from a SecretTen into dst.
    // This guarantees each underlying chunk_id is always accessed with its canonical chunk byte size,
    // avoiding the "same id different size" crash when STORE_CHUNK_ELEM is larger than per-head blocks.
    auto read_slice = [&](shared_ptr<SecretTen> ten, int offset, int len, DtypeForCpuOp* dst, double& get_ms_acc) {
        int copied = 0;
        while (copied < len) {
            int global_idx = offset + copied;
            int chunk_idx = global_idx / STORE_CHUNK_ELEM;
            int chunk_start = chunk_idx * STORE_CHUNK_ELEM;
            int in_chunk_off = global_idx - chunk_start;
            int can_copy = std::min(len - copied, STORE_CHUNK_ELEM - in_chunk_off);
            int chunk_elems = chunk_elems_for(ten, chunk_idx);
            int chunk_bytes = chunk_elems * (int)sizeof(DtypeForCpuOp);

            // Decrypt full chunk (canonical size), then copy subrange
            std::vector<DtypeForCpuOp> tmp(chunk_elems);
            {
                auto t0 = layer_now_us();
                chunk_manager.GetChunk(ten->ChunkIds[chunk_idx], tmp.data(), chunk_bytes);
                auto t1 = layer_now_us();
                get_ms_acc += layer_elapsed_ms(t0, t1);
            }
            memcpy(dst + copied, tmp.data() + in_chunk_off, can_copy * sizeof(DtypeForCpuOp));
            copied += can_copy;
        }
    };

    // Helper: write a contiguous slice [offset, offset+len) (in elements) from src into a SecretTen.
    // We preserve other data in the same chunk by read-modify-write on the canonical chunk size.
    auto write_slice = [&](shared_ptr<SecretTen> ten, int offset, int len, const DtypeForCpuOp* src, double& store_ms_acc) {
        int copied = 0;
        while (copied < len) {
            int global_idx = offset + copied;
            int chunk_idx = global_idx / STORE_CHUNK_ELEM;
            int chunk_start = chunk_idx * STORE_CHUNK_ELEM;
            int in_chunk_off = global_idx - chunk_start;
            int can_copy = std::min(len - copied, STORE_CHUNK_ELEM - in_chunk_off);
            int chunk_elems = chunk_elems_for(ten, chunk_idx);
            int chunk_bytes = chunk_elems * (int)sizeof(DtypeForCpuOp);

            std::vector<DtypeForCpuOp> tmp(chunk_elems);
            // Read existing chunk (canonical size) so we don't clobber unrelated regions.
            {
                auto t0 = layer_now_us();
                chunk_manager.GetChunk(ten->ChunkIds[chunk_idx], tmp.data(), chunk_bytes);
                auto t1 = layer_now_us();
                // Treat output read as part of store path (we need RMW); include in store bucket.
                store_ms_acc += layer_elapsed_ms(t0, t1);
            }
            memcpy(tmp.data() + in_chunk_off, src + copied, can_copy * sizeof(DtypeForCpuOp));
            {
                auto t0 = layer_now_us();
                chunk_manager.StoreChunk(ten->ChunkIds[chunk_idx], tmp.data(), chunk_bytes);
                auto t1 = layer_now_us();
                store_ms_acc += layer_elapsed_ms(t0, t1);
            }
            copied += can_copy;
        }
    };
    
    // Process each (batch, head) pair
    for (int bh = 0; bh < num_batch_head; bh++) {
        int A_offset = bh * size_A;
        int B_offset = bh * size_B;
        int C_offset = bh * size_C;
        
        // Load input matrices
        // Load per-head matrices (robust to any STORE_CHUNK_ELEM)
        read_slice(input1_tensor, A_offset, size_A, A_chunk, get1_ms);
        read_slice(input2_tensor, B_offset, size_B, B_chunk, get2_ms);

        // Perform matrix multiplication
        {
            auto t0 = layer_now_us();
            matmul_single(A_chunk, B_chunk, C_chunk, M, K, N, transpose_a, transpose_b);
            auto t1 = layer_now_us();
            compute_ms += layer_elapsed_ms(t0, t1);
        }
        
        // Store output matrix
        write_slice(output_tensor, C_offset, size_C, C_chunk, store_ms);
    }

    // num_inputs=2, separate get timings for input1/input2 (get_ms + get2_ms)
    SecretSetLayerRuntimeStats(FunId, get1_ms, compute_ms, store_ms, get2_ms, 0.0, 2);
}



