#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <string>

#include "feature_set.h"
#include "feature_transformer.h"
#include "sparse_batch.h"

extern const int L1;
extern const int L2;
extern const int L3;

struct NNUEModelImpl : torch::nn::Module
{
public:
    FeatureSetPy *feature_set;

    // DoubleFeatureTransformerSlice *input;
    FeatureTransformerSliceEmulate input = nullptr;
    //FeatTransSlow input = nullptr;
    torch::nn::Linear l1 = nullptr;
    torch::nn::Linear l2 = nullptr;
    torch::nn::Linear output = nullptr;
    torch::nn::Linear dummy = nullptr;

    std::string description;

    NNUEModelImpl(FeatureSetPy *feature_set_ptr);
    NNUEModelImpl(std::string fn);
    ~NNUEModelImpl();
    torch::Tensor forward(torch::Tensor us,
                          torch::Tensor them,
                          torch::Tensor white_indices,
                          torch::Tensor white_values,
                          torch::Tensor black_indices,
                          torch::Tensor black_values);

    torch::Tensor compute_loss(SparseBatchTensors &batch_tensors,
                               int batch_idx,
                               std::string loss_type); // compute loss
    void get_optimizer();                              // return optimizer pointer; create if not exist

private:
    void zero_virtual_feature_weights();
};
TORCH_MODULE(NNUEModel);

// main function to run training
void train_nnue_model();

#endif // #define MODEL_H
