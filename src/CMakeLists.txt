find_package(PythonInterp REQUIRED)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpp-trainer)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

# Add CUDA files to the project
#set(CUDA_FILES
#    vector_add.cu
#)

# Add C++ files to the project
#set(CPP_FILES
#    main.cpp
#)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -g -G -arch=sm_70;)

add_executable(cpp-trainer main.cpp model.cpp serialize.cpp feature_transformer.cpp sparse_batch.cpp feature_set.cpp ranger.cpp my_logger.cpp featrans_kernel.cu)
target_link_libraries(cpp-trainer "${TORCH_LIBRARIES}")
target_link_libraries(cpp-trainer Threads::Threads)
set_property(TARGET cpp-trainer PROPERTY CXX_STANDARD 17)