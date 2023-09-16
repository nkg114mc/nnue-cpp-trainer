
#pragma once
#ifndef SPARSE_BATCH_H
#define SPARSE_BATCH_H

#include <torch/torch.h>

struct SparseBatch;

struct SparseBatchTensors {
public:

    SparseBatchTensors(SparseBatch *batch);
    SparseBatchTensors(SparseBatch *batch, torch::Device *device);
    void get_tensors();

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

#endif // #define SPARSE_BATCH_H
