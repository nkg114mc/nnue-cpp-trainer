/*
class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('max_active_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int)),
        ('white_values', ctypes.POINTER(ctypes.c_float)),
        ('black_values', ctypes.POINTER(ctypes.c_float))
    ]

    def get_tensors(self, device):
        white_values = torch.from_numpy(np.ctypeslib.as_array(self.white_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_values = torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        white_indices = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_indices = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        return us, them, white_indices, white_values, black_indices, black_values, outcome, score
*/
#include <chrono>

#include "sparse_batch.h"
#include "model.h"
#include "serialize.h"

SparseBatchTensors::SparseBatchTensors(SparseBatch *batch)
{
    this->batch_ptr = batch;
    get_tensors();
}

void SparseBatchTensors::get_tensors()
{
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32); //.dtype(torch::kFloat64).device(torch::kCUDA, 1);
    auto int_options = torch::TensorOptions().dtype(torch::kInt32);
    white_values = torch::from_blob(batch_ptr->white_values, {batch_ptr->size, batch_ptr->max_active_features}, float_options); // torch.from_numpy(np.ctypeslib.as_array(, shape=(self.))).pin_memory().to(device=device, non_blocking=True)
    black_values = torch::from_blob(batch_ptr->black_values, {batch_ptr->size, batch_ptr->max_active_features}, float_options); // torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
    white_indices = torch::from_blob(batch_ptr->white, {batch_ptr->size, batch_ptr->max_active_features}, int_options);         // torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
    black_indices = torch::from_blob(batch_ptr->black, {batch_ptr->size, batch_ptr->max_active_features}, int_options);         // torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
    us = torch::from_blob(batch_ptr->is_white, {batch_ptr->size, 1}, float_options);                                            // torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
    them = 1.0 - us;
    outcome = torch::from_blob(batch_ptr->outcome, {batch_ptr->size, 1}, float_options); // torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
    score = torch::from_blob(batch_ptr->score, {batch_ptr->size, 1}, float_options);     // torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
}

// for testing
void test_read_batch_stream()
{
    FeatureSetPy *fs_ptr;
    auto nnue_model = NNUEModel(fs_ptr);
    auto feature_trans = FeatureTransformerSliceEmulate(41024, 256);

    for (const auto& p : nnue_model->parameters()) {
        std::cout << p.sizes() << std::endl;
    }

    FeatureSetPy fs;
    //load_model_from_nnuebin("./nn-myoutput.nnue", &fs);
    //load_model_from_nnuebin("./nn-62ef826d1a6d.nnue", &fs);
    load_model_from_nnuebin("./output-0.nnue", &fs);

    int batch_size = 10;
    auto stream = create_sparse_batch_stream("HalfKP", 4, "/media/mc/Fastdata/Stockfish-NNUE/validate1m/val_1m_d14.bin", batch_size, true, false, 0, false);
    auto t0 = std::chrono::high_resolution_clock::now();

    std::cout << "batch_size = " << batch_size << '\n';
    for (int i = 0; i < 1000; ++i)
    {
        SparseBatch *batch;
        batch = stream->next();
        std::cout << "batch " << i << '\n';
        std::cout << batch->num_inputs << " " << batch->size << " " << batch->max_active_features << std::endl;
        /*
                int total = batch->size * batch->max_active_features;
                for (int id = 0; id < total; id++) {
                    std::cout << batch->white[id] << " " << std::endl;
                }
        */
        SparseBatchTensors batch_tensors(batch);
        std::cout << "before forward" << std::endl;

        auto output = nnue_model->forward(batch_tensors.us,
                                          batch_tensors.them,
                                          batch_tensors.white_indices,
                                          batch_tensors.white_values,
                                          batch_tensors.black_indices,
                                          batch_tensors.black_values);
        /*
                std::cout << batch_tensors.white_indices.type() << " ";
                std::cout << batch_tensors.white_values.type() << " ";
                std::cout << batch_tensors.black_indices.type() << " ";
                std::cout << batch_tensors.black_values.type() << std::endl;

                std::cout << batch_tensors.white_indices.sizes() << " ";
                std::cout << batch_tensors.white_values.sizes() << " ";
                std::cout << batch_tensors.black_indices.sizes() << " ";
                std::cout << batch_tensors.black_values.sizes() << std::endl;

                auto output = feature_trans->forward(batch_tensors.white_indices,
                                                     batch_tensors.white_values,
                                                     batch_tensors.black_indices,
                                                     batch_tensors.black_values);*/
        std::cout << output.sizes() << std::endl;
        // if (i % 100 == 0) std::cout << i << '\n';

        destroy_sparse_batch(batch);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}

void train_nnue_model() {


    
}