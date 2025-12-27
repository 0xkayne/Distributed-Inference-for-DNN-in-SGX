#ifdef USE_SGX
#include "Enclave.h"
#endif

#include "layernorm.hpp"
#include "chunk_manager.hpp"

#include <cmath>
#include <algorithm>

using namespace std;

LayernormBuffer::LayernormBuffer(IdT FunId_) : FunId(FunId_) {}

void LayernormBuffer::init(
    IdT input, IdT output, IdT gamma, IdT beta,
    uint32_t batch_, uint32_t seq_len_, uint32_t embed_dim_,
    float eps_) {
    
    input_tensor = GetTenById(input);
    output_tensor = GetTenById(output);
    gamma_tensor = GetTenById(gamma);
    beta_tensor = GetTenById(beta);
    
    batch = batch_;
    seq_len = seq_len_;
    embed_dim = embed_dim_;
    epsilon = eps_;
    
    total_tokens = batch * seq_len;
    num_elements = total_tokens * embed_dim;
}

void LayernormBuffer::forward() {
    auto& chunk_manager = TrustedChunkManager::getInstance();
    
    // Allocate chunks for input/output data
    DtypeForCpuOp *data_chunk;
    ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);
    
    // Get gamma and beta parameters (small, fits in single chunk)
    vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>> small_chunks;
    DtypeForCpuOp *gamma_chunk = get_small_chunk(gamma_tensor, small_chunks);
    DtypeForCpuOp *beta_chunk = get_small_chunk(beta_tensor, small_chunks);
    
    // Process each token independently
    // LayerNorm normalizes over the embed_dim dimension
    int chunk_size = std::min(STORE_CHUNK_ELEM, num_elements);
    int tokens_per_chunk = chunk_size / embed_dim;
    
    for (int start_token = 0; start_token < total_tokens; start_token += tokens_per_chunk) {
        int num_tokens_in_chunk = std::min(tokens_per_chunk, total_tokens - start_token);
        int chunk_size_in_byte = num_tokens_in_chunk * embed_dim * sizeof(DtypeForCpuOp);
        int start_elem = start_token * embed_dim;
        
        // Load input chunk
        chunk_manager.GetChunk(input_tensor->GetChunkId(start_elem), data_chunk, chunk_size_in_byte);
        
        // Process each token in this chunk
        for (int t = 0; t < num_tokens_in_chunk; t++) {
            DtypeForCpuOp* token_data = data_chunk + t * embed_dim;
            
            // Compute mean
            DtypeForCpuOp mean = 0.0f;
            for (int i = 0; i < embed_dim; i++) {
                mean += token_data[i];
            }
            mean /= embed_dim;
            
            // Compute variance
            DtypeForCpuOp var = 0.0f;
            for (int i = 0; i < embed_dim; i++) {
                DtypeForCpuOp diff = token_data[i] - mean;
                var += diff * diff;
            }
            var /= embed_dim;
            
            // Normalize and apply affine transformation
            DtypeForCpuOp inv_std = 1.0f / std::sqrt(var + epsilon);
            for (int i = 0; i < embed_dim; i++) {
                token_data[i] = (token_data[i] - mean) * inv_std * gamma_chunk[i] + beta_chunk[i];
            }
        }
        
        // Store output chunk
        chunk_manager.StoreChunk(output_tensor->GetChunkId(start_elem), data_chunk, chunk_size_in_byte);
    }
    
    // Release small chunks
    for (auto& p : small_chunks) {
        p.first->ReturnChunk(p.second);
    }
}

void LayernormBuffer::backward() {
    // Not implemented for inference-only mode
    throw std::runtime_error("LayerNorm backward not implemented");
}


