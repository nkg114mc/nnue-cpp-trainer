#include <torch/script.h>
#include <iostream>

#include "ranger.h"

void test_read_batch_stream();
void train_nnue_model();
void training_speed_benckmark();
//void test_feature_transformer_backward();
void test_model_forward_and_loss();
void test_model_backward();


int main()
{
    //torch::Device device(torch::kCUDA);

    torch::Tensor tensor = torch::eye(3);
    //tensor.to(device);
    std::cout << tensor << std::endl;
    torch::Tensor rnd_grad = torch::rand({3, 3});
    std::cout << "rnd_grad = " << rnd_grad << std::endl;
    
    std::cout << tensor.grad() << std::endl;
    tensor.mutable_grad() = rnd_grad;
    std::cout << tensor.grad() << std::endl;

    // test_construct_feature_transformer();
    //test_read_batch_stream();

    train_nnue_model();
    //training_speed_benckmark();

    //test_create_ranger_optimizer();
    //test_centralized_gradient();
    //test_compute_buffer();
    //test_ranger_step();

    //test_model_forward_and_loss();
    //test_model_backward();
}
