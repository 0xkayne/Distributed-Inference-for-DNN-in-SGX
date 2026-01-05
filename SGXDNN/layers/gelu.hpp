#ifndef GELU_H
#define GELU_H

#ifdef USE_SGX
#include "Enclave.h"
#endif

#include <cstdint>
#include <memory>

#include "common_with_enclaves.h"
#include "secret_tensor.hpp"

using namespace std;
using std::shared_ptr;

/**
 * GELU (Gaussian Error Linear Unit) implementation for SGX Enclave.
 * 
 * Uses the fast tanh approximation:
 * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 * 
 * This is the standard activation used in Transformer FFN layers.
 */
class GELUBuffer {
public:
    GELUBuffer() {}
    GELUBuffer(IdT FunId_);
    
    ~GELUBuffer() = default;
    
    /**
     * Initialize GELU buffer.
     * 
     * @param input Input tensor ID
     * @param output Output tensor ID
     * @param num_elements Total number of elements
     * @param use_approximate Whether to use tanh approximation (faster)
     */
    void init(
        IdT input, IdT output,
        uint32_t num_elements, bool use_approximate);
    
    /**
     * Forward pass for GELU.
     */
    void forward();
    
    IdT FunId;
    int num_elements;
    bool use_approximate;
    
    shared_ptr<SecretTen> input_tensor;
    shared_ptr<SecretTen> output_tensor;
    
    // Constants
    static constexpr DtypeForCpuOp SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/π)
    static constexpr DtypeForCpuOp GELU_COEFF = 0.044715f;
};

#endif // GELU_H



