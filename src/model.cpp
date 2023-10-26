#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include "model.h"

// 3 layer fully connected network
const int L1 = 256;
const int L2 = 32;
const int L3 = 32;

const int INPUT_DIM = 41024;

NNUEModelImpl::NNUEModelImpl(FeatureSetPy *feature_set_ptr)
{
    this->feature_set = feature_set_ptr;

    this->input = DoubleFeatureTransformerSlice(INPUT_DIM, L1);
    //this->input = FeatureTransformerSliceEmulate(INPUT_DIM, L1);
    //this->input = FeatTransSlow(INPUT_DIM, L1);
    register_module("input", this->input);
    this->l1 = register_module("l1", torch::nn::Linear(2 * L1, L2));
    this->l2 = register_module("l2", torch::nn::Linear(L2, L3));
    this->output = register_module("output", torch::nn::Linear(L3, 1));

    this->description = "No description set?";

    zero_virtual_feature_weights();
}


torch::Tensor load_txt_tensor(std::ifstream &inf);

NNUEModelImpl::NNUEModelImpl(std::string fn)
{
    this->input = DoubleFeatureTransformerSlice(INPUT_DIM, L1);
    //this->input = FeatureTransformerSliceEmulate(INPUT_DIM, L1);
    //this->input = FeatTransSlow(INPUT_DIM, L1);
    this->l1 = torch::nn::Linear(2 * L1, L2);
    this->l2 = torch::nn::Linear(L2, L3);
    this->output = torch::nn::Linear(L3, 1);

    std::ifstream inf;
    inf.open(fn);

    this->input->weight = register_parameter("weight2", load_txt_tensor(inf), false);
    this->input->bias = register_parameter("bias2", load_txt_tensor(inf), false);
    this->l1->weight = register_parameter("weight3", load_txt_tensor(inf), false);
    this->l1->bias = register_parameter("bias3", load_txt_tensor(inf), false);
    this->l2->weight = register_parameter("weight4", load_txt_tensor(inf), false);
    this->l2->bias = register_parameter("bias4", load_txt_tensor(inf), false);
    this->output->weight = register_parameter("weight5", load_txt_tensor(inf), false);
    this->output->bias = register_parameter("bias5", load_txt_tensor(inf), false);

    this->input->weight.requires_grad_();
    this->input->bias.requires_grad_();
    this->l1->weight.requires_grad_();
    this->l1->bias.requires_grad_();
    this->l2->weight.requires_grad_();
    this->l2->bias.requires_grad_();
    this->output->weight.requires_grad_();
    this->output->bias.requires_grad_();
    
    inf.close();

    register_module("input", this->input);
    register_module("l1", this->l1);
    register_module("l2", this->l2);
    register_module("output", this->output);

    this->description = "Debug model (parameter initialized)";

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
    //auto wb_pair = input->forward_separate(white_indices, white_values, black_indices, black_values);
    //auto w = wb_pair.first;
    //auto b = wb_pair.second;
    auto ft_outputs = input->forward(white_indices, white_values, black_indices, black_values);
    auto w = ft_outputs[0];
    auto b = ft_outputs[1];
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
torch::Tensor NNUEModelImpl::compute_loss(SparseBatchTensors &batch_tensors,
                                          int batch_idx,
                                          std::string loss_type)
{
    /*
        def step_(self, batch, batch_idx, loss_type):
            us, them, white_indices, white_values, black_indices, black_values, outcome, score = batch

            # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
            # This needs to match the value used in the serializer
            nnue2score = 600
            scaling = 361

            q = self(us, them, white_indices, white_values, black_indices, black_values) * nnue2score / scaling
            p = (score / scaling).sigmoid()

            epsilon = 1e-12
            teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())  # result
            teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))  # entropy
            loss = teacher_loss.mean() - teacher_entropy.mean()
            self.log(loss_type, loss)
            return loss
    */

    // us, them, white_indices, white_values, black_indices, black_values, outcome, score = batch

    // 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    // This needs to match the value used in the serializer
    const double nnue2score = 600;
    const double scaling = 361;

    auto y = this->forward(batch_tensors.us,
                           batch_tensors.them,
                           batch_tensors.white_indices,
                           batch_tensors.white_values,
                           batch_tensors.black_indices,
                           batch_tensors.black_values);
    auto q = y * nnue2score / scaling;
    auto p = (batch_tensors.score / scaling).sigmoid();

    double epsilon = 1e-12;
    //std::cout << epsilon * 1000000 << std::endl;
    auto teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log()); // result
    auto teacher_loss = -(p * torch::nn::functional::logsigmoid(q) + (1.0 - p) * torch::nn::functional::logsigmoid(-q)); // entropy
    auto loss = teacher_loss.mean() - teacher_entropy.mean();
    // self.log(loss_type, loss)

    //std::cout << "Loss: "
    //          << "batch=" << batch_idx << " "
    //          << "loss=" << loss << std::endl;
    return loss;
}

// return optimizer pointer; create if not exist
void NNUEModelImpl::get_optimizer()
{
}
