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
    std::cout << "learning_rate" << " " << input_option.lr() << std::endl;

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


void test_model_forward_and_loss()
{
    FeatureSetPy feat_set;
    //auto nnue_model = NNUEModel(&feat_set);
    auto nnue_model = NNUEModel("/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/nn_params.txt");

    torch::Device main_device(torch::kCUDA);
    nnue_model->to(main_device);

    //for (auto &p : nnue_model->parameters()) {
    //    p.to(main_device);
    //    std::cout << p.sizes() << " " << p.device() << std::endl;
    //}

    int batch_size = 10000;
    std::string fn = "/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/val_1m_d14.bin";
    auto stream = create_sparse_batch_stream("HalfKP", 4, fn.c_str(), batch_size, true, false, 0, false);
    int64_t total_size = 100 * 1000000;
    int64_t batch_cnt = total_size / batch_size;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "batch_size = " << batch_size << '\n';
    for (int batch_id = 0; batch_id < batch_cnt; batch_id++)
    {
        //std::cout << "start batch " << batch_id << '\n';
        SparseBatch *batch;
        batch = stream->next();
        SparseBatchTensors batch_tensors(batch, &main_device);

        auto loss = nnue_model->compute_loss(batch_tensors, batch_id, "some_loss");

        loss.backward();

        std::cout << loss << std::endl;
        destroy_sparse_batch(batch);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}

void test_model_backward()
{
    FeatureSetPy feat_set;
    //auto nnue_model = NNUEModel(&feat_set);
    auto nnue_model = NNUEModel("/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/nn_params.txt");

    std::cout << &(nnue_model->input->weight) << std::endl;
    std::cout << &(nnue_model->input->bias) << std::endl;

    auto optim_option = torch::optim::AdamOptions(0.001);
    torch::optim::Adam optimizer(nnue_model->parameters(), optim_option);

    for (auto &p : nnue_model->parameters()) {
        std::cout << "param: " <<  &(p) << p.sizes() << std::endl;
    }


    int batch_size = 10000;
    std::string fn = "/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/val_1m_d14.bin";
    auto stream = create_sparse_batch_stream("HalfKP", 4, fn.c_str(), batch_size, true, false, 0, false);
    int64_t total_size = 100 * 1000000;
    int64_t batch_cnt = 2; //total_size / batch_size;
    

    std::ifstream inf;
    inf.open("/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/backward_test.txt");

    std::cout << "batch_size = " << batch_size << '\n';
    for (int batch_id = 0; batch_id < batch_cnt; batch_id++)
    {
        //std::cout << "start batch " << batch_id << '\n';
        SparseBatch *batch;
        batch = stream->next();
        SparseBatchTensors batch_tensors(batch);

        optimizer.zero_grad();

/*
        std::cout << nnue_model->input->weight.grad() << std::endl;
        std::cout << nnue_model->input->bias.grad() << std::endl;

        std::cout << nnue_model->l1->weight.grad() << std::endl;
        std::cout << nnue_model->l1->bias.grad() << std::endl;

        std::cout << nnue_model->l2->weight.grad() << std::endl;
        std::cout << nnue_model->l2->bias.grad() << std::endl;

        std::cout << nnue_model->output->weight.grad() << std::endl;
        std::cout << nnue_model->output->bias.grad() << std::endl;
*/

        auto loss = nnue_model->compute_loss(batch_tensors, batch_id, "some_loss");
        std::cout << "loss = " << loss << std::endl;

        loss.backward();

/*
        std::cout << nnue_model->input->weight.grad().norm() << std::endl;
        std::cout << nnue_model->input->bias.grad().norm() << std::endl;

        std::cout << nnue_model->l1->weight.grad() << std::endl;
        std::cout << nnue_model->l1->bias.grad() << std::endl;

        std::cout << nnue_model->l2->weight.grad() << std::endl;
        std::cout << nnue_model->l2->bias.grad() << std::endl;

        std::cout << nnue_model->output->weight.grad() << std::endl;
        std::cout << nnue_model->output->bias.grad() << std::endl;
*/

        // load expected
        auto weight_grad = load_txt_tensor(inf);
        auto bias_grad = load_txt_tensor(inf);
        std::cout << "weight difference norm = " << torch::norm(nnue_model->input->weight.grad() - weight_grad) << std::endl;
        std::cout << "bias difference norm = " << torch::norm(nnue_model->input->bias.grad() - bias_grad) << std::endl;

        auto l1_weight_grad = load_txt_tensor(inf);
        auto l1_bias_grad = load_txt_tensor(inf);
        std::cout << "weight difference norm = " << torch::norm(nnue_model->l1->weight.grad() - l1_weight_grad) << std::endl;
        std::cout << "bias difference norm = " << torch::norm(nnue_model->l1->bias.grad() - l1_bias_grad) << std::endl;

        std::cout << nnue_model->l1->bias.grad() << std::endl;
        std::cout << l1_bias_grad << std::endl;

        auto l2_weight_grad = load_txt_tensor(inf);
        auto l2_bias_grad = load_txt_tensor(inf);
        std::cout << "weight difference norm = " << torch::norm(nnue_model->l2->weight.grad() - l2_weight_grad) << std::endl;
        std::cout << "bias difference norm = " << torch::norm(nnue_model->l2->bias.grad() - l2_bias_grad) << std::endl;

        auto output_weight_grad = load_txt_tensor(inf);
        auto output_bias_grad = load_txt_tensor(inf);
        std::cout << "weight difference norm = " << torch::norm(nnue_model->output->weight.grad() - output_weight_grad) << std::endl;
        std::cout << "bias difference norm = " << torch::norm(nnue_model->output->bias.grad() - output_bias_grad) << std::endl;

        destroy_sparse_batch(batch);
    }

    inf.close();
}


void test_model_params_init()
{
    FeatureSetPy feat_set;
    auto nnue_model = NNUEModel(&feat_set);
    
    std::ifstream inf;
    inf.open("/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/nn_params_pyinit.txt");

    auto weight_py = load_txt_tensor(inf);
    auto bias_py = load_txt_tensor(inf);
    std::cout << "weight difference norm = " << torch::norm(nnue_model->input->weight) << " " << torch::norm(weight_py) << std::endl;
    std::cout << "bias difference norm = " << torch::norm(nnue_model->input->bias) << " " << torch::norm(bias_py) << std::endl;

    //std::cout << nnue_model->input->bias << " " << bias_py << std::endl;

    auto l1_weight_py = load_txt_tensor(inf);
    auto l1_bias_py = load_txt_tensor(inf);
    std::cout << "weight difference norm = " << torch::norm(nnue_model->l1->weight) << " " << torch::norm(l1_weight_py) << std::endl;
    std::cout << "bias difference norm = " << torch::norm(nnue_model->l1->bias) << " " << torch::norm(l1_bias_py) << std::endl;

    auto l2_weight_py = load_txt_tensor(inf);
    auto l2_bias_py = load_txt_tensor(inf);
    std::cout << "weight difference norm = " << torch::norm(nnue_model->l2->weight) << " " << torch::norm(l2_weight_py) << std::endl;
    std::cout << "bias difference norm = " << torch::norm(nnue_model->l2->bias) << " " << torch::norm(l2_bias_py) << std::endl;

    //std::cout << nnue_model->l2->weight << " " << l2_weight_py << std::endl;
    //std::cout << nnue_model->l2->bias << " " << l2_bias_py << std::endl;

    auto output_weight_py = load_txt_tensor(inf);
    auto output_bias_py = load_txt_tensor(inf);
    std::cout << "weight difference norm = " << torch::norm(nnue_model->output->weight) << " " << torch::norm(output_weight_py) << std::endl;
    std::cout << "bias difference norm = " << torch::norm(nnue_model->output->bias) << " " << torch::norm(output_bias_py) << std::endl;

    inf.close();
}


float lweight[41024 * 256];
float lbias[256];
float output[256];

int feature_indices[10 * 32];
float feature_values[10 * 32];

void test_featrans_kernel()
{
    //FeatureSetPy feat_set;
    //auto nnue_model = NNUEModel(&feat_set);

    //read_txt_nnue_model(nnue_model, "/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/nn_params.txt");

    int batch_size = 10;
    /*
    std::string fn = "/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/val_1m_d14.bin";
    auto stream = create_sparse_batch_stream("HalfKP", 1, fn.c_str(), batch_size, true, false, 0, false);
    int64_t total_size = 100 * 1000000;
    int64_t batch_cnt = total_size / batch_size;
*/
    //auto linear = torch::nn::Linear(INPUT_DIM, L1);

    for (int i = 0; i < 10; i++) {
        for (int j = 0 ; j < 32; j++) {
            feature_indices[i * 32 + j] = j;
            feature_values[i * 32 + j] = 1.0;
        }
    }

    feature_transformer_slice_forward_wrapper(10, feature_indices, feature_values, lweight, lbias, output); 
/*
    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "batch_size = " << batch_size << '\n';
    for (int batch_id = 0; batch_id < batch_cnt; batch_id++)
    {
        std::cout << "start batch " << batch_id << '\n';

        SparseBatch *batch;
        batch = stream->next();
        SparseBatchTensors batch_tensors(batch);


        //int num_threads = 256;

        //template <uint32_t output_size, uint32_t max_active_features, uint32_t output_thread_slice_size>
        //feature_transformer_slice_forward<256, 32, 4>(batch->white, batch->white_values, linear->weight.data_ptr(), linear->bias.data_ptr(), output);
        
        //feature_transformer_slice_forward_wrapper(1, batch->white, batch->white_values, lweight, lbias, output); 

        //destroy_sparse_batch(batch);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
*/
}
