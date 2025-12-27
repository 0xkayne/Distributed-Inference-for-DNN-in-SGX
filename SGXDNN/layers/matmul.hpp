#ifndef MATMUL_H
#define MATMUL_H

#ifdef USE_SGX
#include "Enclave.h"
#endif

#include <cstdint>
#include <memory>
#include <vector>

#include "common_with_enclaves.h"
#include "secret_tensor.hpp"

using namespace std;
using std::shared_ptr;

/**
 * Batch Matrix Multiplication for SGX Enclave.
 * 
 * Supports:
 * - Q @ K^T: (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
 * - Attention @ V: (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
 * 
 * Unlike Linear layer, MatMul has no learnable parameters.
 * It operates on two input tensors.
 */
class MatMulBuffer {
public:
    MatMulBuffer() {}
    MatMulBuffer(IdT FunId_);
    
    ~MatMulBuffer() = default;
    
    /**
     * Initialize MatMul buffer.
     * 
     * @param input1 First input tensor ID
     * @param input2 Second input tensor ID
     * @param output Output tensor ID
     * @param batch Batch size
     * @param num_heads Number of attention heads
     * @param seq_len Sequence length
     * @param dim1 Last dimension of first input (after transpose if needed)
     * @param dim2 Last dimension of second input (after transpose if needed)
     * @param transpose_a Whether to transpose first input
     * @param transpose_b Whether to transpose second input
     * @param scale Optional scaling factor
     */
    void init(
        IdT input1, IdT input2, IdT output,
        uint32_t batch, uint32_t num_heads, uint32_t seq_len,
        uint32_t dim1, uint32_t dim2,
        bool transpose_a, bool transpose_b,
        float scale);
    
    /**
     * Forward pass for MatMul.
     */
    void forward();
    
    IdT FunId;
    int batch;
    int num_heads;
    int seq_len;
    int dim1;  // For Q@K^T: head_dim; For Attn@V: seq_len
    int dim2;  // For Q@K^T: seq_len; For Attn@V: head_dim
    bool transpose_a;
    bool transpose_b;
    DtypeForCpuOp scale;
    
    // Computed dimensions
    int M;  // Output rows per head
    int N;  // Output cols per head
    int K;  // Shared dimension
    
    shared_ptr<SecretTen> input1_tensor;
    shared_ptr<SecretTen> input2_tensor;
    shared_ptr<SecretTen> output_tensor;
    
private:
    /**
     * Perform single matrix multiplication: C = A @ B
     */
    void matmul_single(
        DtypeForCpuOp* A, DtypeForCpuOp* B, DtypeForCpuOp* C,
        int M, int K, int N, bool trans_a, bool trans_b);
};

#endif // MATMUL_H


