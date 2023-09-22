#pragma once
#ifndef FEATRANS_KERNEL_H
#define FEATRANS_KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned int uint32_t;
typedef int int32_t;


void feature_transformer_slice_forward_wrapper(
    const int32_t  batch_size,
    const int32_t* const feature_indices,
    const float*   const feature_values,
    const float*   const weight,
    const float*   const bias,
          float*   const output
);

void feature_transformer_slice_backward_wrapper(
    const int32_t  batch_size,
    const int32_t* const feature_indices,
    const float*   const feature_values,
          float*   const weight_grad,
          float*   const bias_grad,
    const float*   const output_grad
);

#endif // #ifndef FEATRANS_KERNEL_H