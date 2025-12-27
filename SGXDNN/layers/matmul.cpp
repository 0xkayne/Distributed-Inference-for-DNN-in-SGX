#ifdef USE_SGX
#include "Enclave.h"
#endif

#include "matmul.hpp"
#include "chunk_manager.hpp"

#include <cmath>
#include <algorithm>
#include <cstring>

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
    
    // Process each (batch, head) pair
    for (int bh = 0; bh < num_batch_head; bh++) {
        int A_offset = bh * size_A;
        int B_offset = bh * size_B;
        int C_offset = bh * size_C;
        
        // Load input matrices
        int A_size_bytes = size_A * sizeof(DtypeForCpuOp);
        int B_size_bytes = size_B * sizeof(DtypeForCpuOp);
        int C_size_bytes = size_C * sizeof(DtypeForCpuOp);
        
        chunk_manager.GetChunk(input1_tensor->GetChunkId(A_offset), A_chunk, A_size_bytes);
        chunk_manager.GetChunk(input2_tensor->GetChunkId(B_offset), B_chunk, B_size_bytes);
        
        // Perform matrix multiplication
        matmul_single(A_chunk, B_chunk, C_chunk, M, K, N, transpose_a, transpose_b);
        
        // Store output matrix
        chunk_manager.StoreChunk(output_tensor->GetChunkId(C_offset), C_chunk, C_size_bytes);
    }
}


