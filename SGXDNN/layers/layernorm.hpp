#ifndef LAYERNORM_H
#define LAYERNORM_H

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
 * LayerNorm implementation for SGX Enclave.
 * 
 * Computes: y = (x - mean) / sqrt(var + eps) * gamma + beta
 * where mean and var are computed over the normalized dimensions (typically last dim).
 * 
 * Unlike BatchNorm which normalizes over batch dimension,
 * LayerNorm normalizes over feature dimensions.
 */
class LayernormBuffer {
public:
    LayernormBuffer() {}
    LayernormBuffer(IdT FunId_);
    
    ~LayernormBuffer() = default;
    
    /**
     * Initialize LayerNorm buffer.
     * 
     * @param input Input tensor ID
     * @param output Output tensor ID
     * @param gamma Scale parameter tensor ID
     * @param beta Bias parameter tensor ID
     * @param batch Batch size
     * @param seq_len Sequence length (number of tokens)
     * @param embed_dim Embedding dimension (normalized dimension)
     * @param eps Small constant for numerical stability
     */
    void init(
        IdT input, IdT output, IdT gamma, IdT beta,
        uint32_t batch, uint32_t seq_len, uint32_t embed_dim,
        float eps);
    
    /**
     * Forward pass for LayerNorm.
     */
    void forward();
    
    /**
     * Backward pass for LayerNorm (not implemented for inference).
     */
    void backward();
    
    IdT FunId;
    int batch;
    int seq_len;
    int embed_dim;
    DtypeForCpuOp epsilon;
    
    // Tensor dimensions
    int total_tokens;  // batch * seq_len
    int num_elements;  // total_tokens * embed_dim
    
    // Tensor pointers
    shared_ptr<SecretTen> input_tensor;
    shared_ptr<SecretTen> output_tensor;
    shared_ptr<SecretTen> gamma_tensor;
    shared_ptr<SecretTen> beta_tensor;
};

#endif // LAYERNORM_H


