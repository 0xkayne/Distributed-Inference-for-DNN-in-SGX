#ifndef SOFTMAX_H
#define SOFTMAX_H

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
 * Softmax implementation for SGX Enclave.
 * 
 * Computes: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
 * The max-subtraction is used for numerical stability.
 * 
 * For Transformer attention: applied along the last dimension (keys)
 * Input shape: (B, H, N, N) for attention weights
 */
class SoftmaxBuffer {
public:
    SoftmaxBuffer() {}
    SoftmaxBuffer(IdT FunId_);
    
    ~SoftmaxBuffer() = default;
    
    /**
     * Initialize Softmax buffer.
     * 
     * @param input Input tensor ID
     * @param output Output tensor ID
     * @param total_elements Total number of elements
     * @param softmax_dim Size of the dimension to apply softmax (last dim)
     */
    void init(
        IdT input, IdT output,
        uint32_t total_elements, uint32_t softmax_dim);
    
    /**
     * Forward pass for Softmax.
     */
    void forward();
    
    IdT FunId;
    int total_elements;
    int softmax_dim;      // Size of the last dimension
    int num_softmax_ops;  // total_elements / softmax_dim
    
    shared_ptr<SecretTen> input_tensor;
    shared_ptr<SecretTen> output_tensor;
};

#endif // SOFTMAX_H



