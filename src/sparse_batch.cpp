#include "training_data_loader.h"
#include "sparse_batch.h"


SparseBatchTensors::SparseBatchTensors(SparseBatch *batch)
{
    this->batch_ptr = batch;
    this->device_ptr = nullptr;
    if (device_ptr == nullptr) {
        std::cout << "Warning: device pointer is nullptr." << std::endl;
    }
    get_tensors();
}

SparseBatchTensors::SparseBatchTensors(SparseBatch *batch, torch::Device *device)
{
    this->batch_ptr = batch;
    this->device_ptr = device;
    get_tensors();
}

SparseBatchTensors::SparseBatchTensors(void *batch_ptr, torch::Device *device)
{
    this->batch_ptr = static_cast<SparseBatch*>(batch_ptr);
    this->device_ptr = device;
    if (device_ptr == nullptr) {
        std::cout << "Warning: device pointer is nullptr." << std::endl;
    }
    get_tensors();
}

void SparseBatchTensors::get_tensors()
{
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32); //.dtype(torch::kFloat64).device(torch::kCUDA, 1);
    auto int_options = torch::TensorOptions().dtype(torch::kInt32);
    //if (device_ptr != nullptr) {
    //    float_options = torch::TensorOptions().dtype(torch::kFloat32).device(*device_ptr);
    //    int_options = torch::TensorOptions().dtype(torch::kInt32).device(*device_ptr);
    //}
    white_values = torch::from_blob(batch_ptr->white_values, {batch_ptr->size, batch_ptr->max_active_features}, float_options).pin_memory().to(*device_ptr); // torch.from_numpy(np.ctypeslib.as_array(, shape=(self.))).pin_memory().to(device=device, non_blocking=True)
    black_values = torch::from_blob(batch_ptr->black_values, {batch_ptr->size, batch_ptr->max_active_features}, float_options).pin_memory().to(*device_ptr); // torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
    white_indices = torch::from_blob(batch_ptr->white, {batch_ptr->size, batch_ptr->max_active_features}, int_options).pin_memory().to(*device_ptr);         // torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
    black_indices = torch::from_blob(batch_ptr->black, {batch_ptr->size, batch_ptr->max_active_features}, int_options).pin_memory().to(*device_ptr);         // torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
    us = torch::from_blob(batch_ptr->is_white, {batch_ptr->size, 1}, float_options).pin_memory().to(*device_ptr);                                            // torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
    them = 1.0 - us;
    outcome = torch::from_blob(batch_ptr->outcome, {batch_ptr->size, 1}, float_options).pin_memory().to(*device_ptr); // torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
    score = torch::from_blob(batch_ptr->score, {batch_ptr->size, 1}, float_options).pin_memory().to(*device_ptr);     // torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
}

void SparseBatchTensors::free_sparse_batch() {
    destroy_sparse_batch(this->batch_ptr);
}


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

SparseBatchStreamWrapper::SparseBatchStreamWrapper() {
    this->stream = nullptr;
}

SparseBatchStreamWrapper::~SparseBatchStreamWrapper() {
    delete this->stream;
}

void SparseBatchStreamWrapper::create(const char* feature_set_c, int concurrency, const char* filename, int batch_size, bool cyclic,
                                      bool filtered, int random_fen_skipping, bool wld_filtered) {
    this->stream = create_sparse_batch_stream(
        feature_set_c,
        concurrency,
        filename, 
        batch_size, 
        cyclic, 
        filtered, 
        random_fen_skipping, 
        wld_filtered
    );
}

void* SparseBatchStreamWrapper::next() {
    return (void*)(stream->next());
}
