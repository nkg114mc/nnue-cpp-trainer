#include <torch/torch.h>
#include <iostream>
#include <memory>

#include "model.h"

// 3 layer fully connected network
const int L1 = 256;
const int L2 = 32;
const int L3 = 32;

const int INPUT_DIM = 41024;

NNUEModelImpl::NNUEModelImpl(FeatureSetPy *feature_set_ptr)
{
    this->feature_set = feature_set_ptr;

    // this->input = register_module<FeatureTransformerSliceEmulate>("input", std::make_shared<FeatureTransformerSliceEmulate>(INPUT_DIM, L1));
    this->input = FeatureTransformerSliceEmulate(INPUT_DIM, L1);
    register_module("input", this->input);
    this->l1 = register_module("l1", torch::nn::Linear(2 * L1, L2));
    this->l2 = register_module("l2", torch::nn::Linear(L2, L3));
    this->output = register_module("output", torch::nn::Linear(L3, 1));

    this->description = "No description set?";

    zero_virtual_feature_weights();
}

NNUEModelImpl::~NNUEModelImpl()
{
}

torch::Tensor NNUEModelImpl::forward(torch::Tensor us,
                                 torch::Tensor them,
                                 torch::Tensor white_indices,
                                 torch::Tensor white_values,
                                 torch::Tensor black_indices,
                                 torch::Tensor black_values)
{
    auto wb_pair = input->forward_separate(white_indices, white_values, black_indices, black_values);
    auto w = wb_pair.first;
    auto b = wb_pair.second;
    auto l0_ = (us * torch::cat({w, b}, 1)) + (them * torch::cat({b, w}, 1));
    // clamp here is used as a clipped relu to (0.0, 1.0)
    auto l0_clamp_ = torch::clamp(l0_, 0.0, 1.0);
    auto l1_ = torch::clamp(l1->forward(l0_clamp_), 0.0, 1.0);
    auto l2_ = torch::clamp(l2->forward(l1_), 0.0, 1.0);
    auto x = output->forward(l2_);
    return x;
}
/*
        w, b = self.input(white_indices, white_values, black_indices, black_values)
        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        # clamp here is used as a clipped relu to (0.0, 1.0)
        l0_ = torch.clamp(l0_, 0.0, 1.0)
        l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
        l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
        x = self.output(l2_)
        return x
*/

void NNUEModelImpl::zero_virtual_feature_weights()
{
}
/*
        self.input = DoubleFeatureTransformerSlice(feature_set.num_features, L1)
        self.feature_set = feature_set
        self.l1 = nn.Linear(2 * L1, L2)
        self.l2 = nn.Linear(L2, L3)
        self.output = nn.Linear(L3, 1)
        self._zero_virtual_feature_weights()
        self.description = "??
*/

// compute loss
torch::Tensor NNUEModelImpl::compute_loss() {
}

// return optimizer pointer; create if not exist
void NNUEModelImpl::get_optimizer() {

}

//void NNUEModelImpl::parameters() {
//
//}
