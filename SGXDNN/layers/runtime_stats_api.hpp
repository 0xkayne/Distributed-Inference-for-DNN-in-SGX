#ifndef SGXDNN_LAYER_RUNTIME_STATS_API_HPP
#define SGXDNN_LAYER_RUNTIME_STATS_API_HPP

#include <cstdint>

#include "common_with_enclaves.h"  // IdT

// Lightweight API for SGXDNN layers to publish per-forward timing breakdowns.
// Times are in milliseconds.
extern "C" {

// num_inputs: 1 for single-input ops (gelu/softmax/layernorm), 2 for matmul, etc.
// get_ms: time spent loading input(s) via TrustedChunkManager::GetChunk (and any other loads the op does)
// compute_ms: time spent in pure compute loops (excluding GetChunk/StoreChunk)
// store_ms: time spent storing outputs via TrustedChunkManager::StoreChunk
// get2_ms/store2_ms: second-input or auxiliary I/O timing (used by MatMul to separate input2 load)
void SecretSetLayerRuntimeStats(
    IdT FunId,
    double get_ms,
    double compute_ms,
    double store_ms,
    double get2_ms,
    double store2_ms,
    int32_t num_inputs);

}

#endif // SGXDNN_LAYER_RUNTIME_STATS_API_HPP


