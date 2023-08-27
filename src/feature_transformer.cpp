#include <cmath>
#include <iostream>
// #include <cuda.h>

#include "feature_transformer.h"

/*
DoubleFeatureTransformerSliceImpl::DoubleFeatureTransformerSliceImpl(int n_inputs, int n_outputs)
{
    this->num_inputs = n_inputs;
    this->num_outputs = n_outputs;

    float sigma = std::sqrt(1.0 / (float)(num_inputs));
    auto init_weight = torch::rand({num_inputs, num_outputs}, torch::kF32) * (2 * sigma) - sigma;
    auto init_bias = torch::rand({num_outputs}, torch::kF32) * (2 * sigma) - sigma;
    this->weight = register_parameter("weight", init_weight, false);
    this->bias = register_parameter("bias", init_bias, false);
}*/

////////////////////////////////////////////////////////////////////////////////

class FeatTransFunctionSlow : public torch::autograd::Function<FeatTransFunctionSlow>
{
public:
    static torch::autograd::tensor_list forward(torch::autograd::AutogradContext *ctx,
                                                torch::Tensor feature_indices_0,
                                                torch::Tensor feature_values_0,
                                                torch::Tensor feature_indices_1,
                                                torch::Tensor feature_values_1,
                                                torch::Tensor weight,
                                                torch::Tensor bias)
    {
        // Save data for backward in context
        // ctx->saved_data["n"] = n;
        // var.mul_(2);
        // Mark var as modified by inplace operation
        // ctx->mark_dirty({var});
        // return {var};

        //std::cout << "-----------> weight require_grad = " << weight.requires_grad_() << std::endl;
        //std::cout << "-----------> weight bias_grad = " << bias.requires_grad_() << std::endl;

        auto output_0 = forward_single(ctx, feature_indices_0, feature_values_0, weight, bias);
        auto output_1 = forward_single(ctx, feature_indices_1, feature_values_1, weight, bias);

        //std::cout << "-----------> output_0 sizes = " << output_0[0].sizes() << std::endl;
        //std::cout << "-----------> output_1 sizes = " << output_1[0].sizes() << std::endl;
        return {output_0[0], output_1[0]};
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::tensor_list grad_outputs)
    {
        auto grad_weight_bias_0 = backward_single(ctx, {grad_outputs[0]});
        auto grad_weight_bias_1 = backward_single(ctx, {grad_outputs[1]});
        auto grad_weight = grad_weight_bias_0[2] + grad_weight_bias_1[2];
        auto grad_bias = grad_weight_bias_0[3] + grad_weight_bias_1[3];

        //std::cout << "===========> grad_weight sizes = " << grad_weight.sizes() << std::endl;
        //std::cout << "===========> grad_bias sizes = " << grad_bias.sizes() << std::endl;
        
        // return {grad_weight, grad_bias};
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), grad_weight, grad_bias};
    }

private:
    static torch::autograd::tensor_list forward_single(torch::autograd::AutogradContext *ctx,
                                                       torch::Tensor feature_indices,
                                                       torch::Tensor feature_values,
                                                       torch::Tensor weight,
                                                       torch::Tensor bias)
    {
        ctx->save_for_backward({feature_indices, feature_values, weight, bias});

        int batch_size = feature_indices.size(0);
        int num_inputs = weight.size(0);
        int max_active_features = feature_indices.size(1);
        auto inputs = torch::zeros({batch_size, num_inputs}, torch::kF32);

        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < max_active_features; j++)
            {
                int feature = feature_indices[i][j].item<int>();
                float value = feature_values[i][j].item<float>();
                inputs[i][feature] += value;
            }
        }

        torch::Tensor y = torch::mm(inputs, weight) + bias;
        return {y};
    }

    static torch::autograd::tensor_list backward_single(torch::autograd::AutogradContext *ctx,
                                                        torch::autograd::tensor_list grad_outputs)
    {
        torch::Tensor grad_output = grad_outputs[0].contiguous();

        auto saved = ctx->get_saved_variables();
        auto feature_indices = saved[0].contiguous();
        auto feature_values = saved[1].contiguous();
        auto weight = saved[2].contiguous();
        auto bias = saved[3].contiguous();

        int batch_size = feature_indices.size(0);
        int num_inputs = weight.size(0);
        int max_active_features = feature_indices.size(1);
        auto inputs_transpose = torch::zeros({num_inputs, batch_size}, torch::kF32);

        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < max_active_features; j++)
            {
                int feature = feature_indices[i][j].item<int>();
                float value = feature_values[i][j].item<float>();
                inputs_transpose[feature][i] += value;
            }
        }

        auto weight_grad = torch::mm(inputs_transpose, grad_output);
        auto bias_grad = grad_output.sum(0); //(grad_output, dim=0)

        return {torch::Tensor(), torch::Tensor(), weight_grad, bias_grad};
    }
};

// wrapper class
FeatTransSlowImpl::FeatTransSlowImpl(int n_inputs, int n_outputs)
{
    this->num_inputs = n_inputs;
    this->num_outputs = n_outputs;

    float sigma = std::sqrt(1.0 / (float)(num_inputs));
    auto init_weight = torch::rand({num_inputs, num_outputs}, torch::kF32) * (2 * sigma) - sigma;
    auto init_bias = torch::rand({num_outputs}, torch::kF32) * (2 * sigma) - sigma;
    this->weight = register_parameter("weight", init_weight, false);
    this->bias = register_parameter("bias", init_bias, false);

    weight.requires_grad_();
    bias.requires_grad_();
}

torch::autograd::tensor_list FeatTransSlowImpl::forward(torch::Tensor feature_indices_0,
                                                        torch::Tensor feature_values_0,
                                                        torch::Tensor feature_indices_1,
                                                        torch::Tensor feature_values_1)
{
    return FeatTransFunctionSlow::apply(feature_indices_0, feature_values_0,
                                        feature_indices_1, feature_values_1,
                                        this->weight, this->bias);
}
////////////////////////////////////////////////////////////////////////////////

torch::Tensor FeatureTransformerSliceFunctionEmulate(torch::Tensor feature_indices,
                                                     torch::Tensor feature_values,
                                                     torch::Tensor weight,
                                                     torch::Tensor bias)
{
    int batch_size = feature_indices.size(0);
    int num_inputs = weight.size(0);
    int max_active_features = feature_indices.size(1);
    auto inputs = torch::zeros({batch_size, num_inputs}, torch::kF32);

    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < max_active_features; j++)
        {
            int feature = feature_indices[i][j].item<int>();
            float value = feature_values[i][j].item<float>();
            inputs[i][feature] += value;
        }
    }

    return torch::mm(inputs, weight) + bias;
}

////////////////////////////////////////////////////////////////////////////////

FeatureTransformerSliceEmulateImpl::FeatureTransformerSliceEmulateImpl(int n_inputs, int n_outputs)
{
    // super(DoubleFeatureTransformerSlice, self).__init__()
    this->num_inputs = n_inputs;
    this->num_outputs = n_outputs;

    float sigma = std::sqrt(1.0 / (float)(num_inputs));
    auto init_weight = torch::rand({num_inputs, num_outputs}, torch::kF32) * (2 * sigma) - sigma;
    auto init_bias = torch::rand({num_outputs}, torch::kF32) * (2 * sigma) - sigma;
    this->weight = register_parameter("weight", init_weight, false);
    this->bias = register_parameter("bias", init_bias, false);

    weight.requires_grad_();
    bias.requires_grad_();
}

torch::autograd::tensor_list FeatureTransformerSliceEmulateImpl::forward(torch::Tensor feature_indices_0,
                                                                         torch::Tensor feature_values_0,
                                                                         torch::Tensor feature_indices_1,
                                                                         torch::Tensor feature_values_1)
{
    /*
            return DoubleFeatureTransformerSliceFunction.apply(feature_indices_0, feature_values_0, feature_indices_1,
                                                               feature_values_1, self.weight, self.bias)
    */
    torch::autograd::tensor_list outputs;
    auto output00 = FeatureTransformerSliceFunctionEmulate(feature_indices_0, feature_values_0, weight, bias);
    auto output01 = FeatureTransformerSliceFunctionEmulate(feature_indices_1, feature_values_1, weight, bias);
    // return torch::cat({output00, output01}, 1);
    outputs.push_back(output00);
    outputs.push_back(output01);
    return outputs;
}

std::pair<torch::Tensor, torch::Tensor> FeatureTransformerSliceEmulateImpl::forward_separate(torch::Tensor feature_indices_0,
                                                                                             torch::Tensor feature_values_0,
                                                                                             torch::Tensor feature_indices_1,
                                                                                             torch::Tensor feature_values_1)
{
    auto output00 = FeatureTransformerSliceFunctionEmulate(feature_indices_0, feature_values_0, weight, bias);
    auto output01 = FeatureTransformerSliceFunctionEmulate(feature_indices_1, feature_values_1, weight, bias);
    return std::make_pair(output00, output01);
}

void test_construct_feature_transformer()
{
    auto feature_trans = FeatureTransformerSliceEmulate(41024, 256);
    std::cout << feature_trans << std::endl;

    int BATCH_SIZE = 16;
    int INPUT_SIZE = 10;
    int MAX_ACTIVE_FEATURES = 32;

    auto indices0 = (torch::rand({BATCH_SIZE, MAX_ACTIVE_FEATURES}) * INPUT_SIZE).to(torch::kInt32);
    auto indices1 = (torch::rand({BATCH_SIZE, MAX_ACTIVE_FEATURES}) * INPUT_SIZE).to(torch::kInt32);
    auto values0 = torch::rand({BATCH_SIZE, MAX_ACTIVE_FEATURES}, torch::kF32);
    auto values1 = torch::rand({BATCH_SIZE, MAX_ACTIVE_FEATURES}, torch::kF32);
    auto outputs = feature_trans->forward(indices0, values0, indices1, values1);
    std::cout << outputs[0].sizes() << std::endl;
    std::cout << outputs[1].sizes() << std::endl;
}
