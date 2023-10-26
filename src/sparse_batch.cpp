#include <chrono>

#include "training_data_loader.h"
#include "sparse_batch.h"
#include "model.h"
#include "serialize.h"
#include "ranger.h"
#include "my_logger.h"

#include "featrans_kernel.cuh"

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void read_txt_nnue_model(NNUEModel &model, std::string fn) {

    std::ifstream inf;
    inf.open(fn);

    model->input->weight = load_txt_tensor(inf);
    model->input->bias = load_txt_tensor(inf);
    model->l1->weight = load_txt_tensor(inf);
    model->l1->bias = load_txt_tensor(inf);
    model->l2->weight = load_txt_tensor(inf);
    model->l2->bias = load_txt_tensor(inf);
    model->output->weight = load_txt_tensor(inf);
    model->output->bias = load_txt_tensor(inf);

    model->input->weight.requires_grad_();
    model->input->bias.requires_grad_();
    model->l1->weight.requires_grad_();
    model->l1->bias.requires_grad_();
    model->l2->weight.requires_grad_();
    model->l2->bias.requires_grad_();
    model->output->weight.requires_grad_();
    model->output->bias.requires_grad_();

    inf.close();

    std::cout << "Done reading model parameters from txt" << std::endl;
}

void run_validation(std::string val_fn, int64_t epoch_size, torch::Device &main_device, NNUEModel &model, MyFileLogger &trlogger) {

    int random_fen_skipping = 3;
    int64_t batch_size = 10000;
    auto stream = create_sparse_batch_stream("HalfKP", 32, val_fn.c_str(), batch_size, true, true, random_fen_skipping, false);
    int64_t batch_cnt = (epoch_size + batch_size - 1) / batch_size;

    for (int batch_id = 0; batch_id < batch_cnt; batch_id++)
    {
        std::cout << "Running validation batch " << batch_id << '\n';

        SparseBatch *batch;
        batch = stream->next();
        SparseBatchTensors batch_tensors(batch, &main_device);

        auto loss = model->compute_loss(batch_tensors, batch_id, "val_loss");
            
        trlogger.log_txt("Loss: batch=" + std::to_string(batch_id) + " loss=" + std::to_string(loss.item<float>()) + "\n", false);
        destroy_sparse_batch(batch);
    }
}

void train_nnue_model()
{
    FeatureSetPy feat_set;
    auto nnue_model = NNUEModel(&feat_set);
    //auto nnue_model = NNUEModel("/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/nn_params.txt");
    //auto nnue_model = NNUEModel("/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/nn_params_pyinit.txt");
    

/*
    // torch::optim::SGD optimizer(nnue_model->parameters(), torch::optim::SGDOptions(learning_rate));
    const double learning_rate = 0.0001;
    const double weight_decay = 0.00001;
    auto optim_option = torch::optim::AdamOptions(learning_rate);
    optim_option.weight_decay(weight_decay);
    torch::optim::Adam optimizer(nnue_model->parameters(), optim_option);
*/
    for (auto &p : nnue_model->parameters()) {
        std::cout << "param: " <<  &(p) << p.sizes() << std::endl;
    }
    std::cout << "input param count: " << nnue_model->input->parameters().size() << std::endl;
    std::cout << "l1 param count: " << nnue_model->input->parameters().size() << std::endl;
    std::cout << "l2 param count: " << nnue_model->input->parameters().size() << std::endl;
    std::cout << "output param count: " << nnue_model->input->parameters().size() << std::endl;


    auto ranger_option = RangerOptions().betas({0.9, 0.999}).eps(1.0e-7);
    
    //std::vector<torch::Tensor> input_params{nnue_model->input->parameters()};
    //std::vector<torch::Tensor> hidden_params;
    //for (auto &p : nnue_model->l1->parameters()) {
    //    hidden_params.push_back(p);
    //}
    //for (auto &p : nnue_model->l2->parameters()) {
    //    hidden_params.push_back(p);
    //}
    //std::vector<torch::Tensor> output_params{nnue_model->output->parameters()};
    std::vector<torch::Tensor> input_params{nnue_model->input->weight, nnue_model->input->bias};
    std::vector<torch::Tensor> hidden_params{nnue_model->l1->weight, nnue_model->l1->bias, nnue_model->l2->weight, nnue_model->l2->bias};
    std::vector<torch::Tensor> output_params{nnue_model->output->weight, nnue_model->output->bias};

    RangerOptions input_option(ranger_option);
    RangerOptions hidden_option(ranger_option);
    RangerOptions output_option(ranger_option);
    //LR = 1e-3
    //train_params = [
    //    {'params': self.get_layers(lambda x: self.input == x), 'lr': LR, 'gc_dim': 0},
    //    {'params': self.get_layers(lambda x: self.output != x and self.input != x), 'lr': LR},
    //    {'params': self.get_layers(lambda x: self.output == x), 'lr': LR / 10},
    //]
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

    auto lr_schedulr = torch::optim::StepLR(optimizer, 1, 0.992);

    //torch::Device main_device(torch::kCPU);
    torch::Device main_device(torch::kCUDA);
    nnue_model->to(main_device);

    std::string output_model_fn = "/home/mc/sidework/nnchess/nnue-trainer/my-cpp-trainer/cpp-networks/cpp_output_1.nnue";
    MyFileLogger trlogger("./training_output.log");

    //int64_t batch_size = 8192;
    int64_t batch_size = 16384;
    int random_fen_skipping = 3;
    //std::string train_fn = "/media/mc/Fastdata/Stockfish-NNUE/trainingdata100m/trn_100m_d10.bin";
    //std::string train_fn = "/media/mc/Fastdata/Stockfish-NNUE/trainingdata1b/trn_1b_d10.bin";
    std::string train_fn = "/media/mc/Fastdata/Stockfish-NNUE/training_sf_official/large_gensfen_multipvdiff_100_d9.binpack";
    auto stream = create_sparse_batch_stream("HalfKP", 32, train_fn.c_str(), batch_size, true, true, random_fen_skipping, false);
    //int64_t total_size = 1000000000ll;
    int64_t epoch_size = 100000000ll;
    int64_t batch_cnt = (epoch_size + batch_size - 1) / batch_size;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::cout << "batch_size = " << batch_size << '\n';
    int64_t iter_id = -1;
    int MAX_EPOCH = 600;
    for (int epoch = 0; epoch < MAX_EPOCH; epoch++)
    {
        for (int batch_id = 0; batch_id < batch_cnt; batch_id++)
        {
            iter_id++;

            SparseBatch *batch;
            //std::cout << "start "
            //          << "Epoch " << epoch << " batch " << batch_id << " iter " << iter_id << '\n';
            batch = stream->next();

            SparseBatchTensors batch_tensors(batch, &main_device);

/*
            std::cout << batch_tensors.us.device() << std::endl;
            std::cout << batch_tensors.them.device() << std::endl;
            std::cout << batch_tensors.white_indices.device() << std::endl;
            std::cout << batch_tensors.white_values.device() << std::endl;
            std::cout << batch_tensors.black_indices.device() << std::endl;
            std::cout << batch_tensors.black_values.device() << std::endl;
*/

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
            
            if (iter_id % 10 == 0) {
                trlogger.log_txt("Loss: epoch=" + std::to_string(epoch) + " batch=" + std::to_string(iter_id) + " loss=" + std::to_string(loss.item<float>()) + "\n", true);
            }
            
            //std::cout << nnue_model->input->bias.grad() << std::endl;
            //std::cout << nnue_model->l1->bias.grad() << std::endl;
            loss.backward();
            //std::cout << nnue_model->input->bias.grad().sizes() << std::endl;
            //std::cout << nnue_model->input->weight.grad().sizes() << std::endl;
            //std::cout << nnue_model->l1->bias.grad().sizes() << std::endl;

/*
            std::cout << nnue_model->l1->weight.grad() << std::endl;
            std::cout << nnue_model->l1->bias.grad() << std::endl;

            std::cout << nnue_model->l2->weight.grad() << std::endl;
            std::cout << nnue_model->l2->bias.grad() << std::endl;

            std::cout << nnue_model->output->weight.grad() << std::endl;
            std::cout << nnue_model->output->bias.grad() << std::endl;
*/

            optimizer.step();

            destroy_sparse_batch(batch);
        }

        // reduce learning rate
        lr_schedulr.step();

        std::cout << optimizer.param_groups()[0].options().get_lr() << std::endl;
        std::cout << optimizer.param_groups()[1].options().get_lr() << std::endl;
        std::cout << optimizer.param_groups()[2].options().get_lr() << std::endl;
    }

    // dump output to model file
    torch::Device cpu_device(torch::kCPU);
    nnue_model->to(cpu_device);
    nnue_model->description = std::string("小严霜单打独根草");
    save_model_nnue_format(nnue_model, output_model_fn);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}

void training_speed_benckmark()
{
    FeatureSetPy feat_set;
    auto nnue_model = NNUEModel(&feat_set);

    auto optim_option = torch::optim::AdamOptions(0.001);
    torch::optim::Adam optimizer(nnue_model->parameters(), optim_option);


    torch::Device cuda_device(torch::kCUDA);
    //torch::Device cuda_device(torch::kCPU);
    nnue_model->to(cuda_device);

    int batch_size = 100000; //10000;
    //std::string train_fn = "/media/mc/Fastdata/Stockfish-NNUE/trainingdata100m/trn_100m_d10.bin";
    //std::string train_fn = "/media/mc/Fastdata/Stockfish-NNUE/trainingdata1b/trn_1b_d10.bin";
    std::string train_fn = "/media/mc/Fastdata/Stockfish-NNUE/training_sf_official/large_gensfen_multipvdiff_100_d9.binpack";
    auto stream = create_sparse_batch_stream("HalfKP", 32, train_fn.c_str(), batch_size, true, false, 0, false);
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

        auto iter_t3 = std::chrono::system_clock::now();

        SparseBatchTensors batch_tensors(batch, &cuda_device);

        auto iter_t2 = std::chrono::system_clock::now();

        optimizer.zero_grad();

        auto loss = nnue_model->compute_loss(batch_tensors, iter_id, "some_loss");

        loss.backward();

        optimizer.step();

        destroy_sparse_batch(batch);

        auto iter_t1 = std::chrono::system_clock::now();
        std::cout << "iteration time: " << (iter_t1 - iter_t0).count() / 1e9 << "s\n";
        std::cout << "data-loading time: " << (iter_t2 - iter_t0).count() / 1e9 << "s\n";
        std::cout << "data-reading time: " << (iter_t3 - iter_t0).count() / 1e9 << "s\n";
        std::cout << "data-convert time: " << (iter_t2 - iter_t3).count() / 1e9 << "s\n";
        std::cout << "optimization time: " << (iter_t1 - iter_t2).count() / 1e9 << "s\n";
    }

    auto t1 = std::chrono::system_clock::now();
    std::cout << "total time: " << (t1 - t0).count() / 1e9 << "s\n";
}
