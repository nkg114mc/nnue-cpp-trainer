
#pragma once
#ifndef SPARSE_BATCH_H
#define SPARSE_BATCH_H

#include <torch/torch.h>

struct SparseBatch;
template <typename T> struct Stream;

struct SparseBatchTensors {
public:

    SparseBatchTensors(SparseBatch *batch);
    SparseBatchTensors(SparseBatch *batch, torch::Device *device);
    SparseBatchTensors(void *batch_ptr, torch::Device *device);
    void get_tensors();

    void free_sparse_batch();

    torch::Tensor white_values;
    torch::Tensor black_values;
    torch::Tensor white_indices;
    torch::Tensor black_indices;
    torch::Tensor us;
    torch::Tensor them;
    torch::Tensor outcome;
    torch::Tensor score;

private:

    SparseBatch *batch_ptr;
    torch::Device *device_ptr;
};

struct SparseBatchStreamWrapper {
public:

    SparseBatchStreamWrapper();
    ~SparseBatchStreamWrapper();
    void create(const char* feature_set_c, int concurrency, const char* filename, int batch_size, bool cyclic,
                bool filtered, int random_fen_skipping, bool wld_filtered);
    void* next();

private:

    Stream<SparseBatch> *stream;
};

#endif // #define SPARSE_BATCH_H
