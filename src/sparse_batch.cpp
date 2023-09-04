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

#include "training_data_loader.h"
#include "sparse_batch.h"
#include "model.h"
#include "serialize.h"
#include "ranger.h"

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

    for (const auto &p : nnue_model->parameters())
    {
        std::cout << p.sizes() << std::endl;
    }

    FeatureSetPy fs;
    // load_model_from_nnuebin("./nn-myoutput.nnue", &fs);
    // load_model_from_nnuebin("./nn-62ef826d1a6d.nnue", &fs);
    // load_model_from_nnuebin("./output-0.nnue", &fs);
    load_model_from_nnuebin("/home/mc/sidework/chessengines/Stockfish-sf_132/src/nn-62ef826d1a6d.nnue", &fs);

    int batch_size = 10000;
    auto stream = create_sparse_batch_stream("HalfKP", 4, "/media/mc/Fastdata/Stockfish-NNUE/validate1m/val_1m_d14.bin", batch_size, false, false, 0, false);
    auto t0 = std::chrono::high_resolution_clock::now();

    std::cout << "batch_size = " << batch_size << '\n';
    for (int i = 0; i < 101; ++i)
    {
        SparseBatch *batch;
        std::cout << "start batch " << i << '\n';
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
        /*
                auto output = nnue_model->forward(batch_tensors.us,
                                                  batch_tensors.them,
                                                  batch_tensors.white_indices,
                                                  batch_tensors.white_values,
                                                  batch_tensors.black_indices,
                                                  batch_tensors.black_values);

                std::cout << output.sizes() << std::endl;
                // if (i % 100 == 0) std::cout << i << '\n';
        */
        destroy_sparse_batch(batch);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}

void configure_ranger_optimizer(NNUEModel &model, RangerOptions &default_option) {

    std::vector<torch::Tensor> input_params{model->input->parameters()};
    std::vector<torch::Tensor> hidden_params;
    for (auto &p : model->l1->parameters()) {
        hidden_params.push_back(p);
    }
    for (auto &p : model->l2->parameters()) {
        hidden_params.push_back(p);
    }
    std::vector<torch::Tensor> output_params{model->output->parameters()};

    auto input_option = RangerOptions(default_option);
    auto hidden_option = RangerOptions(default_option);
    auto output_option = RangerOptions(default_option);

    std::cout << (&default_option) << " " << (&input_option) << std::endl;
    std::cout << "learning_reate" << " " << input_option.lr() << std::endl;

    std::cout << model->input->parameters().size() << " " << input_params.size() << std::endl;
    std::cout << model->input->parameters().size() << " " << output_params.size() << std::endl;
    std::cout << model->l1->parameters().size() << "+" << model->l2->parameters().size() << " " << hidden_params.size() << std::endl;
    /*
        LR = 1e-3
        train_params = [
            {'params': self.get_layers(lambda x: self.input == x), 'lr': LR, 'gc_dim': 0},
            {'params': self.get_layers(lambda x: self.output != x and self.input != x), 'lr': LR},
            {'params': self.get_layers(lambda x: self.output == x), 'lr': LR / 10},
        ]
    */
    const double LR = 1e-3;
    input_option.lr(LR).gc_dim(0);
    hidden_option.lr(LR);
    output_option.lr(LR / 10.0);

/*
    result.push_back(torch::optim::OptimizerParamGroup(input_params, std::make_unique<torch::optim::OptimizerOptions>(input_option)));
    result.push_back(torch::optim::OptimizerParamGroup(hidden_params, std::make_unique<torch::optim::OptimizerOptions>(hidden_option)));
    result.push_back(torch::optim::OptimizerParamGroup(output_params, std::make_unique<torch::optim::OptimizerOptions>(output_option)));
*/
}

void train_nnue_model()
{
    FeatureSetPy feat_set;
    auto nnue_model = NNUEModel(&feat_set);
    // torch::optim::SGD optimizer(nnue_model->parameters(), torch::optim::SGDOptions(learning_rate));
/*
    const double learning_rate = 0.0001;
    const double weight_decay = 0.00001;
    auto optim_option = torch::optim::AdamOptions(learning_rate);
    optim_option.weight_decay(weight_decay);
    torch::optim::Adam adam_optimizer(nnue_model->parameters(), optim_option);
*/
    auto ranger_option = RangerOptions().betas({0.9, 0.999}).eps(1.0e-7);
    
    std::vector<torch::Tensor> input_params{nnue_model->input->parameters()};
    std::vector<torch::Tensor> hidden_params;
    for (auto &p : nnue_model->l1->parameters()) {
        hidden_params.push_back(p);
    }
    for (auto &p : nnue_model->l2->parameters()) {
        hidden_params.push_back(p);
    }
    std::vector<torch::Tensor> output_params{nnue_model->output->parameters()};

    RangerOptions input_option(ranger_option);
    RangerOptions hidden_option(ranger_option);
    RangerOptions output_option(ranger_option);
    /*
        LR = 1e-3
        train_params = [
            {'params': self.get_layers(lambda x: self.input == x), 'lr': LR, 'gc_dim': 0},
            {'params': self.get_layers(lambda x: self.output != x and self.input != x), 'lr': LR},
            {'params': self.get_layers(lambda x: self.output == x), 'lr': LR / 10},
        ]
    */
    const double LR = 1e-3;
    input_option.lr(LR).gc_dim(0);
    hidden_option.lr(LR);
    output_option.lr(LR / 10.0);
    auto g1 = torch::optim::OptimizerParamGroup(input_params);
    auto g2 = torch::optim::OptimizerParamGroup(hidden_params);
    auto g3 = torch::optim::OptimizerParamGroup(output_params);
    g1.set_options(std::make_unique<RangerOptions>(input_option));
    g2.set_options(std::make_unique<RangerOptions>(hidden_option));
    g3.set_options(std::make_unique<RangerOptions>(output_option));
    Ranger optimizer({g1, g2, g3}, ranger_option);
    //Ranger optimizer({torch::optim::OptimizerParamGroup(nnue_model->parameters())}, ranger_option);


    int batch_size = 10000;
    auto stream = create_sparse_batch_stream("HalfKP", 4, "/media/mc/Fastdata/Stockfish-NNUE/trainingdata100m/trn_100m_d10.bin", batch_size, true, false, 0, false);
    int64_t total_size = 100 * 1000000;
    int64_t batch_cnt = total_size / batch_size;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::cout << "batch_size = " << batch_size << '\n';
    int iter_id = -1;
    for (int epoch = 0; epoch <= 10; epoch++)
    {
        for (int batch_id = 0; batch_id < batch_cnt; batch_id++)
        {
            iter_id++;

            SparseBatch *batch;
            std::cout << "start "
                      << "Epoch " << epoch << " batch " << batch_id << " iter " << iter_id << '\n';
            batch = stream->next();

            SparseBatchTensors batch_tensors(batch);
            // std::cout << "before forward" << std::endl;
            /*
                        auto output = nnue_model->forward(batch_tensors.us,
                                                          batch_tensors.them,
                                                          batch_tensors.white_indices,
                                                          batch_tensors.white_values,
                                                          batch_tensors.black_indices,
                                                          batch_tensors.black_values);
            */
            // std::cout << "output = " <<  output << std::endl;
            optimizer.zero_grad();

            auto loss = nnue_model->compute_loss(batch_tensors, iter_id, "some_loss");

            //std::cout << nnue_model->input->bias.grad() << std::endl;
            //std::cout << nnue_model->l1->bias.grad() << std::endl;
            loss.backward();
            //std::cout << nnue_model->input->bias.grad().sizes() << std::endl;
            //std::cout << nnue_model->input->weight.grad().sizes() << std::endl;
            //std::cout << nnue_model->l1->bias.grad().sizes() << std::endl;
            optimizer.step();

            destroy_sparse_batch(batch);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}

void training_speed_benckmark()
{
    FeatureSetPy feat_set;
    auto nnue_model = NNUEModel(&feat_set);

    auto optim_option = torch::optim::AdamOptions(0.001);
    torch::optim::Adam optimizer(nnue_model->parameters(), optim_option);

    int batch_size = 10000;
    auto stream = create_sparse_batch_stream("HalfKP", 4, "/media/mc/Fastdata/Stockfish-NNUE/trainingdata100m/trn_100m_d10.bin", batch_size, true, false, 0, false);
    int64_t total_size = 100 * 1000000;
    int64_t batch_cnt = total_size / batch_size;

    std::cout << "batch_size = " << batch_size << '\n';
    int iter_id = -1;

    auto t0 = std::chrono::system_clock::now();
    for (int batch_id = 0; batch_id < batch_cnt; batch_id++)
    {
        auto iter_t0 = std::chrono::system_clock::now();
        iter_id++;

        SparseBatch *batch;
        std::cout << "start " << "batch " << batch_id << " iter " << iter_id << '\n';
        batch = stream->next();

        SparseBatchTensors batch_tensors(batch);

        optimizer.zero_grad();

        auto loss = nnue_model->compute_loss(batch_tensors, iter_id, "some_loss");

        loss.backward();

        optimizer.step();

        destroy_sparse_batch(batch);

        auto iter_t1 = std::chrono::system_clock::now();
        std::cout << "iteration time: " << (iter_t1 - iter_t0).count() / 1e9 << "s\n";
    }

    auto t1 = std::chrono::system_clock::now();
    std::cout << "total time: " << (t1 - t0).count() / 1e9 << "s\n";
}
