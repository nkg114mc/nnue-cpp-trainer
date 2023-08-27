#pragma once
#ifndef FEATURE_TRANSFORMER_H
#define FEATURE_TRANSFORMER_H

#include <torch/torch.h>
#include <utility>

struct FeatureTransformerSliceEmulateImpl : torch::nn::Module
{
public:
    FeatureTransformerSliceEmulateImpl(int num_inputs, int num_outputs);
    // torch::Tensor forward(torch::Tensor feature_indices_0,
    torch::autograd::tensor_list forward(torch::Tensor feature_indices_0,
                                         torch::Tensor feature_values_0,
                                         torch::Tensor feature_indices_1,
                                         torch::Tensor feature_values_1);

    std::pair<torch::Tensor, torch::Tensor> forward_separate(torch::Tensor feature_indices_0,
                                                             torch::Tensor feature_values_0,
                                                             torch::Tensor feature_indices_1,
                                                             torch::Tensor feature_values_1);

    int num_inputs;
    int num_outputs;
    torch::Tensor weight;
    torch::Tensor bias;
};
TORCH_MODULE(FeatureTransformerSliceEmulate);

/*
// the fast version with cuda optimization
struct DoubleFeatureTransformerSliceImpl : torch::nn::Module
{
public:
    DoubleFeatureTransformerSliceImpl(int num_inputs, int num_outputs);
    torch::Tensor forward(torch::Tensor feature_indices_0,
                          torch::Tensor feature_values_0,
                          torch::Tensor feature_indices_1,
                          torch::Tensor feature_values_1);

    int num_inputs;
    int num_outputs;
    torch::Tensor weight;
    torch::Tensor bias;
};
TORCH_MODULE(DoubleFeatureTransformerSlice);
*/

// slow CPU version
struct FeatTransSlowImpl : torch::nn::Module
{
public:
    FeatTransSlowImpl(int num_inputs, int num_outputs);
    torch::autograd::tensor_list forward(torch::Tensor feature_indices_0,
                                         torch::Tensor feature_values_0,
                                         torch::Tensor feature_indices_1,
                                         torch::Tensor feature_values_1);

    int num_inputs;
    int num_outputs;
    torch::Tensor weight;
    torch::Tensor bias;
};
TORCH_MODULE(FeatTransSlow);

void test_construct_feature_transformer();
void test_feature_transformer_backward();

#endif // #define FEATURE_TRANSFORMER_H