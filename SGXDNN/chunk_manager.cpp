#define USE_EIGEN_TENSOR

#ifndef USE_SGX
#define EIGEN_USE_THREADS
#include <malloc.h>
#else
#include "Enclave.h"
#include "sgx_tseal.h"
#include "sgx_trts.h"
#include "sgx_thread.h"
#endif

#include "chunk_manager.hpp"
#include "common_utils.h"
#include "../App/randpool_common.cpp"
// #include "../App/common_utils.cpp"

using namespace std;

using std::shared_ptr;
using std::make_shared;
using std::unordered_map;
using std::string;
using defer = shared_ptr<void>;




ChunkPool::ChunkPool(int size_pool_, int num_byte_chunk_):
    size_pool(size_pool_), num_byte_chunk(num_byte_chunk_), reserved_base(nullptr), use_edmm(false)
{
    // Initialize committed status vector
    committed.resize(size_pool, false);
    
#ifdef USE_SGX
    // Try to use EDMM if available
    auto& edmm_mgr = EdmmManager::getInstance();
    if (EdmmManager::is_edmm_available()) {
        // Calculate total size needed (aligned to page boundary)
        size_t total_size = (size_t)size_pool * (size_t)num_byte_chunk;
        
        // Reserve memory region using EDMM
        reserved_base = edmm_mgr.reserve_memory(total_size);
        
        if (reserved_base != nullptr) {
            use_edmm = true;
            #ifdef PRINT_CHUNK_INFO
                printf("ChunkPool: Using EDMM with reserved base %p\n", reserved_base);
                printf("Pool size %d, num_byte_chunk %d, total: %lu bytes\n", 
                       size_pool, num_byte_chunk, total_size);
            #endif
            
            // Initialize chunk pointers (offsets from base)
            for (int i = 0; i < size_pool; i++) {
                void* chunk_addr = (char*)reserved_base + ((size_t)i * (size_t)num_byte_chunk);
                chunks.push_back(chunk_addr);
                chunk_ids.push(i);
            }
        } else {
            #ifdef PRINT_CHUNK_INFO
                printf("ChunkPool: EDMM reserve failed, falling back to memalign\n");
            #endif
            use_edmm = false;
        }
    }
#endif

    // Fallback to traditional memalign if EDMM not available or failed
    if (!use_edmm) {
        #ifdef PRINT_CHUNK_INFO
            printf("ChunkPool: Using traditional memalign\n");
            printf("Pool size %d, num_byte_chunk %d\n", size_pool, num_byte_chunk);
        #endif
        
        for (int i = 0; i < size_pool; i++) {
            void* enc_chunk = (void*)memalign(64, num_byte_chunk);
            chunks.push_back(enc_chunk);
            chunk_ids.push(i);
        }
    }
}

ChunkPool::~ChunkPool() {
    if (use_edmm && reserved_base != nullptr) {
#ifdef USE_SGX
        auto& edmm_mgr = EdmmManager::getInstance();
        
        // Decommit all committed pages before freeing
        for (int i = 0; i < size_pool; i++) {
            if (committed[i]) {
                void* chunk_addr = chunks[i];
                edmm_mgr.decommit_pages(chunk_addr, num_byte_chunk);
            }
        }
        
        // Free reserved memory
        edmm_mgr.free_reserved_memory(reserved_base);
        
        #ifdef PRINT_CHUNK_INFO
            printf("ChunkPool: Released EDMM reserved memory\n");
        #endif
#endif
    } else {
        // Free traditional malloc'd chunks
        for (void* chunk : chunks) {
            if (chunk != nullptr) {
                free(chunk);
            }
        }
    }
}

int ChunkPool::get_chunk_id() {
    std::unique_lock<std::mutex> lock(stack_mutex);
    if (chunk_ids.empty()) {
        printf("Running out of chunks\n");
        throw std::invalid_argument("Running out of chunks");
    }
    int res;
    res = chunk_ids.top();
    chunk_ids.pop();
    
    // If using EDMM and chunk not yet committed, commit it now
    if (use_edmm && !committed[res]) {
#ifdef USE_SGX
        auto& edmm_mgr = EdmmManager::getInstance();
        void* chunk_addr = chunks[res];
        
        if (edmm_mgr.commit_pages(chunk_addr, num_byte_chunk)) {
            committed[res] = true;
            #ifdef PRINT_CHUNK_INFO
                printf("ChunkPool: Committed chunk %d at %p (%d bytes)\n", 
                       res, chunk_addr, num_byte_chunk);
            #endif
        } else {
            // Commit failed, put the chunk back
            chunk_ids.push(res);
            lock.unlock();
            printf("ERROR: Failed to commit EDMM pages for chunk %d\n", res);
            throw std::runtime_error("Failed to commit EDMM pages");
        }
#endif
    }
    
    return res;
}

void ChunkPool::return_chunk_id(int id) {
    std::unique_lock<std::mutex> lock(stack_mutex);
    chunk_ids.push(id);
    
    // For EDMM, we could optionally decommit pages here to save EPC
    // For now, we keep them committed for performance (lazy decommit)
    // Uncomment below to enable aggressive decommit on return
    /*
    if (use_edmm && committed[id]) {
#ifdef USE_SGX
        auto& edmm_mgr = EdmmManager::getInstance();
        void* chunk_addr = chunks[id];
        
        if (edmm_mgr.decommit_pages(chunk_addr, num_byte_chunk)) {
            committed[id] = false;
            #ifdef PRINT_CHUNK_INFO
                printf("ChunkPool: Decommitted chunk %d at %p\n", id, chunk_addr);
            #endif
        }
#endif
    }
    */
}


TrustedChunkManager& TrustedChunkManager::getInstance() {
    static TrustedChunkManager instance;
    return instance;
}

IdT TrustedChunkManager::GetNewId() {
    return id_counter++;
}

void TrustedChunkManager::StoreChunk(IdT id, void* src_chunk, int num_byte) {
    int num_byte_enc_chunk = CalcEncDataSize(0, num_byte);
    #ifdef PRINT_CHUNK_INFO
        printf("num in byte %d, ", num_byte);
    #endif
    SgxEncT* enc_chunk = (SgxEncT*) get_untrusted_mem(id, num_byte_enc_chunk);
    DtypeForCpuOp* src_float = (DtypeForCpuOp*) src_chunk;
    encrypt((uint8_t *) src_chunk,
            num_byte,
            (uint8_t *) (&(enc_chunk->payload)),
            (sgx_aes_gcm_128bit_iv_t *)(&(enc_chunk->reserved)),
            (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)));
    // DtypeForCpuOp* dst_chunk = (DtypeForCpuOp*)malloc(num_byte);
    // GetChunk(id, dst_chunk, num_byte);
    // uint8_t* blind_chunk;
    // ChunkGuard<uint8_t> guard(blind_chunks, blind_chunk);
    // decrypt((uint8_t *) (&(enc_chunk->payload)),
    //         num_byte,
    //         (uint8_t *) dst_chunk,
    //         (sgx_aes_gcm_128bit_iv_t  *)(&(enc_chunk->reserved)),
    //         (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)),
    //         (uint8_t *) blind_chunk);
    // src_float = (DtypeForCpuOp*) dst_chunk;
    // free(dst_chunk);
}

void TrustedChunkManager::GetChunk(IdT id, void* dst_chunk, int num_byte) {
    #ifdef PRINT_CHUNK_INFO
        printf("GetChunk, id %ld, num byte %d\n", id, num_byte);
    #endif
    int num_byte_enc_chunk = CalcEncDataSize(0, num_byte);
    uint8_t* blind_chunk;
    ChunkGuard<uint8_t> guard(blind_chunks, blind_chunk);
    SgxEncT* enc_chunk = (SgxEncT*) get_untrusted_mem(id, num_byte_enc_chunk);
    decrypt((uint8_t *) (&(enc_chunk->payload)),
            num_byte,
            (uint8_t *) dst_chunk,
            (sgx_aes_gcm_128bit_iv_t  *)(&(enc_chunk->reserved)),
            (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)),
            (uint8_t *) blind_chunk);
    DtypeForCpuOp* src_float = (DtypeForCpuOp*) dst_chunk;
}

TrustedChunkManager::TrustedChunkManager() {
    max_num_byte_plain_chunk = STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp);
    max_num_byte_enc_chunk = CalcEncDataSize(0, max_num_byte_plain_chunk);

    blind_chunks = make_shared<ChunkPool>(THREAD_POOL_SIZE, max_num_byte_plain_chunk);
}

void* TrustedChunkManager::get_untrusted_mem(IdT id, int num_byte) {
    void* dst_buf;
    bool is_diff_size = false;
    auto it = untrusted_mem_holder.begin();
    auto end = untrusted_mem_holder.end();
    int prev_num_byte;
    {
        std::unique_lock <std::mutex> lock(address_mutex);
        it = untrusted_mem_holder.find(id);
        end = untrusted_mem_holder.end();
    }
    if (it == end) {
        #ifdef PRINT_CHUNK_INFO
            printf("alloc new mem id %u byte %d\n", id, num_byte);
        #endif
        allocate_in_untrusted(&dst_buf, num_byte);
        {
            std::unique_lock<std::mutex> lock(address_mutex);
            untrusted_mem_holder[id] = std::make_pair(dst_buf, num_byte);
        }
    } else {
        std::unique_lock<std::mutex> lock(address_mutex);
        std::tie(dst_buf, prev_num_byte) = untrusted_mem_holder[id];
        if (prev_num_byte != num_byte) {
            is_diff_size = true;
        }
    }
    if (is_diff_size) {
        // Usually cause by passing length instead of num_byte, * sizeof(DtypeForCpuOp)
        printf("id=%u\n",id);
        printf("A id has assigned with multiple size: original: %d, now: %d\n", prev_num_byte, num_byte);
        throw std::invalid_argument("A id has assigned with multiple size.");
    }
    return dst_buf;
}

DtypeForCpuOp* get_small_chunk(
        shared_ptr<SecretTen> tensor,
        vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>>& small_chunks) {

    int size_in_byte = tensor->GetSizeInByte();
    DtypeForCpuOp* arr = (DtypeForCpuOp*) memalign(64, size_in_byte);
    auto& chunk_manager = TrustedChunkManager::getInstance();
    chunk_manager.GetChunk(tensor->GetChunkId(0), arr, size_in_byte);
    small_chunks.emplace_back(tensor, arr);
    return arr;
}

void store_small_chunks(vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>>& small_chunks) {
    for (auto& x : small_chunks) {
        auto tensor = x.first;
        auto arr = x.second;
        auto& chunk_manager = TrustedChunkManager::getInstance();
        int size_in_byte = tensor->GetSizeInByte();
        chunk_manager.StoreChunk(tensor->GetChunkId(0), arr, size_in_byte);
        free(arr);
    }
}
