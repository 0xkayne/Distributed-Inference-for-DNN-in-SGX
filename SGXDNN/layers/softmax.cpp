#ifdef USE_SGX
#include "Enclave.h"
#endif

#include "softmax.hpp"
#include "chunk_manager.hpp"
#include "layer_timing.hpp"
#include "runtime_stats_api.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

using namespace std;

SoftmaxBuffer::SoftmaxBuffer(IdT FunId_) : FunId(FunId_) {}

void SoftmaxBuffer::init(
    IdT input, IdT output,
    uint32_t total_elements_, uint32_t softmax_dim_) {
    
    input_tensor = GetTenById(input);
    output_tensor = GetTenById(output);
    
    total_elements = total_elements_;
    softmax_dim = softmax_dim_;
    num_softmax_ops = total_elements / softmax_dim;
}

void SoftmaxBuffer::forward() {
    auto& chunk_manager = TrustedChunkManager::getInstance();
    
    DtypeForCpuOp *data_chunk;
    ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);

    double get_ms = 0.0;
    double compute_ms = 0.0;
    double store_ms = 0.0;
    
    // Process in chunks, ensuring each chunk contains complete softmax rows
    int chunk_size = std::min(STORE_CHUNK_ELEM, total_elements);
    int rows_per_chunk = chunk_size / softmax_dim;

    // Guard against invalid chunk config (would cause infinite loop)
    if (rows_per_chunk <= 0) {
        // Softmax requires each chunk to contain at least one full row.
        // If STORE_CHUNK_ELEM < softmax_dim, we cannot proceed safely.
        throw std::runtime_error("SoftmaxBuffer: STORE_CHUNK_ELEM too small for softmax_dim");
    }
    
    for (int start_row = 0; start_row < num_softmax_ops; start_row += rows_per_chunk) {
        int num_rows_in_chunk = std::min(rows_per_chunk, num_softmax_ops - start_row);
        int chunk_size_in_byte = num_rows_in_chunk * softmax_dim * sizeof(DtypeForCpuOp);
        int start_elem = start_row * softmax_dim;
        
        // Load input chunk
        {
            auto t0 = layer_now_us();
            chunk_manager.GetChunk(input_tensor->GetChunkId(start_elem), data_chunk, chunk_size_in_byte);
            auto t1 = layer_now_us();
            get_ms += layer_elapsed_ms(t0, t1);
        }
        
        // Apply softmax to each row
        {
            auto t0 = layer_now_us();
            for (int r = 0; r < num_rows_in_chunk; r++) {
                DtypeForCpuOp* row_data = data_chunk + r * softmax_dim;
                
                // Find max for numerical stability
                DtypeForCpuOp max_val = -std::numeric_limits<DtypeForCpuOp>::infinity();
                for (int i = 0; i < softmax_dim; i++) {
                    if (row_data[i] > max_val) {
                        max_val = row_data[i];
                    }
                }
                
                // Compute exp(x - max) and sum
                DtypeForCpuOp sum = 0.0f;
                for (int i = 0; i < softmax_dim; i++) {
                    row_data[i] = std::exp(row_data[i] - max_val);
                    sum += row_data[i];
                }
                
                // Normalize
                DtypeForCpuOp inv_sum = 1.0f / sum;
                for (int i = 0; i < softmax_dim; i++) {
                    row_data[i] *= inv_sum;
                }
            }
            auto t1 = layer_now_us();
            compute_ms += layer_elapsed_ms(t0, t1);
        }
        
        // Store output chunk
        {
            auto t0 = layer_now_us();
            chunk_manager.StoreChunk(output_tensor->GetChunkId(start_elem), data_chunk, chunk_size_in_byte);
            auto t1 = layer_now_us();
            store_ms += layer_elapsed_ms(t0, t1);
        }
    }

    SecretSetLayerRuntimeStats(FunId, get_ms, compute_ms, store_ms, 0.0, 0.0, 1);
}



