#ifdef USE_SGX
#include "Enclave.h"
#endif

#include "gelu.hpp"
#include "chunk_manager.hpp"

#include <cmath>
#include <algorithm>

using namespace std;

GELUBuffer::GELUBuffer(IdT FunId_) : FunId(FunId_) {}

void GELUBuffer::init(
    IdT input, IdT output,
    uint32_t num_elements_, bool use_approximate_) {
    
    input_tensor = GetTenById(input);
    output_tensor = GetTenById(output);
    
    num_elements = num_elements_;
    use_approximate = use_approximate_;
}

/**
 * Fast tanh approximation using polynomial.
 * Accurate for |x| < 5, which covers typical neural network activations.
 */
static inline DtypeForCpuOp fast_tanh(DtypeForCpuOp x) {
    // Clamp to avoid overflow
    if (x > 5.0f) return 1.0f;
    if (x < -5.0f) return -1.0f;
    
    // Use standard tanh for now (can be optimized with polynomial)
    return std::tanh(x);
}

/**
 * Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
static inline DtypeForCpuOp gelu_approximate(DtypeForCpuOp x) {
    DtypeForCpuOp x_cubed = x * x * x;
    DtypeForCpuOp inner = GELUBuffer::SQRT_2_OVER_PI * (x + GELUBuffer::GELU_COEFF * x_cubed);
    return 0.5f * x * (1.0f + fast_tanh(inner));
}

/**
 * Exact GELU: x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
 */
static inline DtypeForCpuOp gelu_exact(DtypeForCpuOp x) {
    static constexpr DtypeForCpuOp INV_SQRT2 = 0.7071067811865476f;  // 1/sqrt(2)
    return x * 0.5f * (1.0f + std::erf(x * INV_SQRT2));
}

void GELUBuffer::forward() {
    auto& chunk_manager = TrustedChunkManager::getInstance();
    
    DtypeForCpuOp *data_chunk;
    ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);
    
    run_all_chunks([&](int start_elem, int num_elem_in_chunk) {
        int chunk_size_in_byte = num_elem_in_chunk * sizeof(DtypeForCpuOp);
        
        // Load input chunk
        chunk_manager.GetChunk(input_tensor->GetChunkId(start_elem), data_chunk, chunk_size_in_byte);
        
        // Apply GELU element-wise
        if (use_approximate) {
            for (int i = 0; i < num_elem_in_chunk; i++) {
                data_chunk[i] = gelu_approximate(data_chunk[i]);
            }
        } else {
            for (int i = 0; i < num_elem_in_chunk; i++) {
                data_chunk[i] = gelu_exact(data_chunk[i]);
            }
        }
        
        // Store output chunk
        chunk_manager.StoreChunk(output_tensor->GetChunkId(start_elem), data_chunk, chunk_size_in_byte);
    }, STORE_CHUNK_ELEM, num_elements);
}


