#ifndef SGXDNN_LAYER_TIMING_HPP
#define SGXDNN_LAYER_TIMING_HPP

// Minimal timing helpers for SGXDNN layer-level profiling.
// We intentionally do NOT include Enclave_t.h here to avoid linkage conflicts
// with other OCALL declarations (e.g., EDMM wrapper headers).

#ifdef USE_SGX
#include "Enclave.h"

extern "C" sgx_status_t ocall_get_time(double* res);

static inline double layer_now_us() {
    double t = 0.0;
    ocall_get_time(&t);
    return t;  // microseconds since epoch
}

static inline double layer_elapsed_ms(double start_us, double end_us) {
    return (end_us - start_us) / 1000.0;
}
#else
#include <chrono>

static inline std::chrono::time_point<std::chrono::high_resolution_clock> layer_now_us() {
    return std::chrono::high_resolution_clock::now();
}

static inline double layer_elapsed_ms(
        std::chrono::time_point<std::chrono::high_resolution_clock> start,
        std::chrono::time_point<std::chrono::high_resolution_clock> end) {
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count() * 1000.0;
}
#endif

#endif // SGXDNN_LAYER_TIMING_HPP


