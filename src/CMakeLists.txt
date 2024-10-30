find_package(PythonInterp REQUIRED)

include_directories("/home/mc/sidework/trytensorbd/tensorboard_logger-master/include")
link_directories("/home/mc/sidework/trytensorbd/tensorboard_logger-master/build")

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpp-trainer)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -g -G -arch=sm_70;)

add_executable(cpp-trainer main.cpp model.cpp serialize.cpp feature_transformer.cpp sparse_batch.cpp feature_set.cpp ranger.cpp train.cpp my_logger.cpp featrans_kernel.cu)
target_link_libraries(cpp-trainer "${TORCH_LIBRARIES}")
target_link_libraries(cpp-trainer Threads::Threads)
target_link_libraries(cpp-trainer tensorboard_logger protobuf)
set_property(TARGET cpp-trainer PROPERTY CXX_STANDARD 17)