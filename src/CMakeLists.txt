cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpp-trainer)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)


add_executable(cpp-trainer main.cpp model.cpp serialize.cpp feature_transformer.cpp sparse_batch.cpp feature_set.cpp)
target_link_libraries(cpp-trainer "${TORCH_LIBRARIES}")
target_link_libraries(cpp-trainer Threads::Threads)
set_property(TARGET cpp-trainer PROPERTY CXX_STANDARD 17)