#include <torch/script.h>
#include <iostream>

void test_read_batch_stream();


int main()
{
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;

    // test_construct_feature_transformer();
    test_read_batch_stream();
}
