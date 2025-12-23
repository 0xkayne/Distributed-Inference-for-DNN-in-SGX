#pragma once

#include "sgx_tseal.h"

// #include "aes_stream_common.hpp"
#include "randpool_common.hpp"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#ifndef UINT32_MAX
#define UINT32_MAX      0xffffffffU
#endif

#define IdT uint64_t
#define EidT uint64_t
#define DtypeForCpuOp float
#define DtypeForEncry uint8_t
#define SgxEncT sgx_aes_gcm_data_t
#define TaskIdT int64_t
#define THREAD_POOL_SIZE 4
// #define NumElemInShard (1 << 21) // 8MB / 4bytes
// #define PrimeLimit ((1 << 16) - 15)
// #define PrimeLimit ((1 << 20) - 3)
// #define PrimeLimit ((1 << 20) - 3)
#define PrimeLimit ((1 << 21) - 9)
// #define PrimeLimit ((1 << 8) - 5)
// #define PrimeLimit ((1 << 12) - 3)
// #define PrimeLimit ((1 << 9) - 3)
#define HalfPrime PrimeLimit / 2

//#define STORE_CHUNK_ELEM 940800
// #define STORE_CHUNK_ELEM 940800
// #define WORK_CHUNK_ELEM 4096
// #define STORE_CHUNK_ELEM 940800
// #define WORK_CHUNK_ELEM 64
// 301056 = 4704 * 64
// If set too small (56*56) input maybe incorrect
// #define STORE_CHUNK_ELEM 940800
// #define STORE_CHUNK_ELEM 940800
// 4704 = 7 * 7 * 3 * 32
// 4704 = 49 * 96
// #define STORE_CHUNK_ELEM 940800
// #define WORK_CHUNK_ELEM 4704
// #define STORE_CHUNK_ELEM 940800
// #define WORK_CHUNK_ELEM 30191616

// 301056
// Original value for small models: 9216
// For standard Inception V3 (299x299 input), need larger value to satisfy divisibility constraints:
// - MaxPool constraints: 1225 (35x35), 289 (17x17), 64 (8x8)
// - Linear constraint: 2048
// Calculated value: 725043200 (satisfies all critical constraints)
// Note: This is ~2.7GB, ensure sufficient EPC memory
// If memory is insufficient, consider using smaller input size (e.g., 224x224) or a smaller value
#define STORE_CHUNK_ELEM 940800
#define WORK_CHUNK_ELEM 725043200

#define NEAREST 0
#define STOCHASTIC 1
#define QUANTIZE_MODE STOCHASTIC

// It should not be hard coded in production
const sgx_aes_gcm_128bit_key_t DataAESKey = {127, 236, 132, 152, 126, 139, 220, 197, 204, 40, 231, 163, 206, 29, 86, 137};

typedef struct DimsStruct {
    int dim0, dim1, dim2, dim3;
} DimsT;

typedef struct EncDimsStruct {
    int TotalNumElem, NumBatches, NumRowsInLastShard, NumRowsInShard, NumRows, NumCols;
} EncDimsT;

typedef struct HandleForEnclaveStruct {
    EncDimsT         EncDims;
} HandleForEnclaveT;

typedef struct HandleForPrgStruct {
    aes_stream_state PrgState;
} HandleForPrgT;

#define MatRowMajor Eigen::Matrix<DtypeForCpuOp, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
#define MapMatRowMajor Eigen::Map<MatRowMajor>
#define EigenTensor Eigen::Tensor<DtypeForCpuOp, 4, Eigen::RowMajor>
// #define EigenTensor Eigen::Tensor<DtypeForCpuOp, 4>Eigen::TensorMap<EigenTensor>
#define MapEigenTensor Eigen::TensorMap<EigenTensor>
#define ConstEigenMatrixMap Eigen::Map<const Eigen::Matrix<DtypeForCpuOp, Eigen::Dynamic, Eigen::Dynamic>>
#define EigenMatrixMap Eigen::Map<Eigen::Matrix<DtypeForCpuOp, Eigen::Dynamic, Eigen::Dynamic>>
#define MapEigenTensor Eigen::TensorMap<EigenTensor>
#define MapEigenVector Eigen::Map<Eigen::RowVectorXf>
