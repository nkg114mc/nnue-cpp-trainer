#include <torch/torch.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <cstring>

#include "model.h"

// hardcoded for now
const uint32_t VERSION = 0x7AF32F16;

/*
def ascii_hist(name, x, bins=6):
    N, X = numpy.histogram(x, bins=bins)
    total = 1.0 * len(x)
    width = 50
    nmax = N.max()

    print(name)
    for (xi, n) in zip(X, N):
        bar = '#' * int(n * 1.0 * width / nmax)
        xi = '{0: <8.4g}'.format(xi).ljust(10)
        print('{0}| {1}'.format(xi, bar))
*/

class NNUEWriter {
public:
    // All values are stored in little endian.

    static uint32_t fc_hash(NNUEModel &model) {
        /*
        // InputSlice hash
        uint32_t prev_hash = 0xEC42E90D;
        prev_hash ^= (L1 * 2);

        // Fully connected layers
        vector<torch::nn::Linear> layers{model.l1, model.l2, model.output}
        for (layer in layers) {
            layer_hash = 0xCC03DAE4;
            layer_hash += layer.out_features
            layer_hash ^= prev_hash >> 1
            layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
            if (layer.out_features != 1) {
                // Clipped ReLU hash
                layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
            }
            prev_hash = layer_hash
        }
        return layer_hash;*/
        return 1664315734;
    }

    NNUEWriter(NNUEModel &md, std::string fn) {
        this->model = md;
        outf.open(fn, std::ios::out | std::ios::binary);
    }

    ~NNUEWriter() {
        outf.close();
    }

    void write_model() {
        uint32_t fc_hash_value = fc_hash(model);
        write_header(model, fc_hash_value);
        write_int32(model->feature_set->get_hash() ^ (L1 * 2));  // Feature transformer hash
        write_feature_transformer(model);
        write_int32(fc_hash_value);  // FC layers hash
        write_fc_layer(model->l1, false);
        write_fc_layer(model->l2, false);
        write_fc_layer(model->output, true);
    }

private:
    std::ofstream outf;
    NNUEModel model{nullptr};

    void write_header(NNUEModel &model, uint32_t fc_hash_value) {
        write_int32(VERSION);  // version
        write_int32(fc_hash_value ^ model->feature_set->get_hash() ^ (L1 * 2));  // halfkp network hash
        //# description = b"Features=HalfKP(Friend)[41024->256x2],"
        //# description += b"Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32]"
        //# description += b"(ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
        // Network definition
        int desc_length = strlen(model->description.c_str());
        write_int32(desc_length);
        outf.write(model->description.c_str(), desc_length);
        std::cout << "description:" << model->description << std::endl;
    }
/*
    void coalesce_ft_weights(NNUEModel &model, layer) {
        weight = layer.weight.data
        indices = model.feature_set.get_virtual_to_real_features_gather_indices()
        weight_coalesced = weight.new_zeros((model.feature_set.num_real_features, weight.shape[1]))
        for i_real, is_virtual in enumerate(indices):
            weight_coalesced[i_real, :] = sum(weight[i_virtual, :] for i_virtual in is_virtual)
        return weight_coalesced
    }
*/
    void write_feature_transformer(NNUEModel &model) {
        // int16 bias = round(x * 127)
        // int16 weight = round(x * 127)
        //layer = model->input;
        //bias = layer.bias.data
        auto bias_int = model->input->bias.mul(127).round().to(torch::kInt16);
        // ascii_hist('ft bias:', bias.numpy())
        int bias_size = bias_int.size(0);
        outf.write(reinterpret_cast<char*>(bias_int.data_ptr<int16_t>()), sizeof(int16_t) * bias_size);
        //self.buf.extend(bias.flatten().numpy().tobytes())

        //weight = self.coalesce_ft_weights(model, layer)
        auto weight_int = model->input->weight.mul(127).round().to(torch::kInt16);
        // ascii_hist('ft weight:', weight.numpy())
        // weights stored as [41024][256]
        int weight_size = weight_int.size(0) * weight_int.size(1);
        outf.write(reinterpret_cast<char*>(weight_int.data_ptr<int16_t>()), sizeof(int16_t) * weight_size);
        //self.buf.extend(weight.flatten().numpy().tobytes())
    }


    void write_fc_layer(torch::nn::Linear &layer, bool is_output) {
        // FC layers are stored as int8 weights, and int32 biases
        int kWeightScaleBits = 6;
        float kActivationScale = 127.0;
        float kBiasScale = 0;
        if (is_output) {
            kBiasScale = (1 << kWeightScaleBits) * kActivationScale;  // = 8128
        } else {
            kBiasScale = 9600.0;  // kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
        }
        float kWeightScale = kBiasScale / kActivationScale;  // = 64.0 for normal layers
        float kMaxWeight = 127.0 / kWeightScale;  // roughly 2.0

        // int32 bias = round(x * kBiasScale)
        // int8 weight = round(x * kWeightScale)
        //bias = layer.bias.data
        auto bias_int = layer->bias.mul(kBiasScale).round().to(torch::kInt32);
        //ascii_hist('fc bias:', bias.numpy())
        //self.buf.extend(bias.flatten().numpy().tobytes())
        int bias_size = bias_int.size(0);
        outf.write(reinterpret_cast<char*>(bias_int.data_ptr<int32_t>()), sizeof(int32_t) * bias_size);

        //clipped = torch::count_nonzero(layer->weight.clamp(-kMaxWeight, kMaxWeight) - layer->weight)
        //total_elements = torch.numel(weight)
        //clipped_max = torch.max(torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight))
        //printf("layer has %d/{} clipped weights. Exceeding by {} the maximum {}.".format(clipped, total_elements, clipped_max, kMaxWeight))
        auto weight_int = layer->weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch::kInt8);
        //ascii_hist('fc weight:', weight.numpy())
        /*
        // FC inputs are padded to 32 elements for simd.
        num_input = weight.shape[1]
        if num_input % 32 != 0:
            num_input += 32 - (num_input % 32)
            new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
            new_w[:, :weight.shape[1]] = weight
            weight = new_w
        */
        // Stored as [outputs][inputs], so we can flatten
        //self.buf.extend(weight.flatten().numpy().tobytes())
        int weight_size = weight_int.size(0) * weight_int.size(1);
        outf.write(reinterpret_cast<char*>(weight_int.data_ptr<int8_t>()), sizeof(int8_t) * weight_size);
    }

    void write_int32(uint32_t v) {
        outf.write(reinterpret_cast<char*>(&v), sizeof(v));
    }

};

class NNUEReader {
public:
    std::ifstream inf;
    FeatureSetPy *feature_set;
    NNUEModel model{nullptr};

    NNUEReader(std::string fn, FeatureSetPy *feature_set) {
        this->inf.open(fn, std::ios::in | std::ios::binary);
        this->feature_set = feature_set;
    }

    ~NNUEReader() {
        this->inf.close();
        //delete model;
    }

    bool read_model() {
        this->model = NNUEModel(feature_set);
        uint32_t fc_hash = NNUEWriter::fc_hash(this->model);

        read_header(feature_set, fc_hash);
        read_int32(feature_set->get_hash() ^ (L1 * 2));  // Feature transformer hash
        read_feature_transformer(model->input);
        read_int32(fc_hash);  // FC layers hash
        read_fc_layer(model->l1, false);
        read_fc_layer(model->l2, false);
        read_fc_layer(model->output, true);

        return inf && inf.peek() == std::ios::traits_type::eof();
    }

private:

    void read_header(FeatureSetPy *feature_set, uint32_t fc_hash) {
        read_int32(VERSION);  // version
        std::cout << feature_set->get_hash() << std::endl;
        read_int32(fc_hash ^ feature_set->get_hash() ^ (L1 * 2));  // halfkp network hash
        int32_t desc_len = read_int32(0);  // Network definition
        char* description_chars = new char[desc_len + 2];
        inf.read(description_chars, desc_len);
        description_chars[desc_len] = '\0';
        model->description = std::string(description_chars);
        delete description_chars;

        std::cout << "Model description: [" << model->description << "]" << std::endl;
    }

    uint32_t get_total_size(const torch::IntArrayRef &shape) {
        if (shape.size() == 0) {
            return 0;
        }
        uint32_t length = shape[0];
        for (int i = 1; i < shape.size(); i++) {
            length *= shape[i];
        }
        std::cout << "length = " << length << std::endl;
        return length;
    }

    torch::Tensor read_tensor(const torch::Dtype &dtype, const torch::IntArrayRef &shape) {
        //d = numpy.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
        //d = torch.from_numpy(d.astype(numpy.float32))
        //d = d.reshape(shape)
        //return d
        auto options = torch::TensorOptions().dtype(dtype);
        //torch::Tensor tensor = torch::full(shape, -1, options);
        uint32_t length = get_total_size(shape);

        char *tmp_data = new char[torch::elementSize(dtype) * length + 1];
        
        std::cout << "before read " << (torch::elementSize(dtype) * length) << std::endl;
        inf.read(tmp_data, torch::elementSize(dtype) * length);
        std::cout << "before from blob" << std::endl;
        auto tensor = torch::from_blob(tmp_data, shape, options);
        tensor.contiguous();
        delete tmp_data;
        return tensor;
    }

    void read_feature_transformer(DoubleFeatureTransformerSlice &layer) {
    //void read_feature_transformer(FeatureTransformerSliceEmulate &layer) {
    //void read_feature_transformer(FeatTransSlow &layer) {
        layer->bias = read_tensor(torch::kInt16, layer->bias.sizes()).divide(127.0);
        // weights stored as [41024][256]
        auto weights_int = read_tensor(torch::kInt16, layer->weight.sizes());
        layer->weight = weights_int.divide(127.0);

        std::cout << "ft bias = " << layer->bias << std::endl;
    }

    void read_fc_layer(torch::nn::Linear &layer, bool is_output) {
        // FC layers are stored as int8 weights, and int32 biases
        int kWeightScaleBits = 6;
        double kActivationScale = 127.0;
        double kBiasScale = 1.0;
        if (!is_output) {
            kBiasScale = (1 << kWeightScaleBits) * kActivationScale;  // = 8128
        } else {
            kBiasScale = 9600.0;  // kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
        }
        double kWeightScale = kBiasScale / kActivationScale;  // = 64.0 for normal layers

        // FC inputs are padded to 32 elements for simd.
        auto non_padded_shape = layer->weight.sizes();
        auto padded_shape = std::vector<int64_t>{non_padded_shape[0], ((non_padded_shape[1] + 31) / 32) * 32};

        std::cout << "non_padded_shape = " << layer->weight.sizes() << std::endl;
        std::cout << "padded_shape = " << padded_shape[1] << std::endl;

        layer->bias = read_tensor(torch::kInt32, layer->bias.sizes()).divide(kBiasScale);
        auto padded_weight = read_tensor(torch::kInt8, padded_shape).divide(kWeightScale);

        // Strip padding.
        //layer.weight.data = layer.weight.data[:non_padded_shape[0], :non_padded_shape[1]]

        std::cout << layer->bias << std::endl;
    }

    uint32_t read_int32(uint32_t expected) {
        uint32_t v;
        bool sizeOk = (bool)inf.read(reinterpret_cast<char*>(&v), sizeof(v));
        if (!sizeOk) {
            std::cerr << "Read int32 failed" << std::endl;
        }
        return v;
    }
};

NNUEModel load_model_from_nnuebin(std::string source_path,
                             FeatureSetPy *feature_set) {
    NNUEReader reader(source_path, feature_set);
    bool read_ok = reader.read_model();
    if (!read_ok) {
        std::cout << "Read error" << std::endl;
    } else {
        std::cout << "Read OK" << std::endl;
    }
    NNUEModel nnue_model = reader.model;
}

/*
void load_model_from_file(std:string source_path, feature_set, model_ptr) {

    # feature_set = features.get_feature_set_from_name(feature_set_name)

    if source_path.endswith('.ckpt'):
        nnue_model = M.NNUE.load_from_checkpoint(source_path, feature_set=feature_set)
        nnue_model.eval()
    elif source_path.endswith('.pt'):
        nnue_model = torch.load(source_path)
    elif source_path.endswith('.nnue'):
        with open(source_path, 'rb') as f:
            reader = NNUEReader(f, feature_set)
            nnue_model = reader.model
    else:
        raise Exception('Invalid network input format.')

    print('Reading model file %s' % source_path)
    return nnue_model
}


void save_model_nnue_format(nnue_model, target_fn) {

    if not target_fn.endswith('.nnue'):
        raise Exception('Invalid network output format.')
    writer = NNUEWriter(nnue_model)
    with open(target_fn, 'wb') as f:
        f.write(writer.buf)

}
*/

void save_model_nnue_format(NNUEModel &nnue_model, std::string target_fn) {
    //if not target_fn.endswith('.nnue'):
    //    raise Exception('Invalid network output format.')
    NNUEWriter writer(nnue_model, target_fn);
    writer.write_model();
}

void test_model_serializer_write() {
    FeatureSetPy feat_set;
    auto nnue_model = NNUEModel("/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/nn_params.txt");
    nnue_model->feature_set = &feat_set;
    //nnue_model->description = "12345shangshandalaohu";
    nnue_model->description = "八月秋风阵阵凉，一场白露一场霜";
    save_model_nnue_format(nnue_model, "/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/cpp-model_output.nnue");
}

void main_simple() {
/*
    source_fn = '/home/mc/sidework/chessengines/Stockfish-sf_132/src/nn-62ef826d1a6d.nnue'
    target_fn = 'nn-mynoutput.nnue'
    feature_set_name = 'HalfKP'

    feature_set = features.get_feature_set_from_name(feature_set_name)

    nnue = None

    print('Converting %s to %s' % (source_fn, target_fn))

    with open(source_fn, 'rb') as f:
        reader = NNUEReader(f, feature_set)
        nnue = reader.model
    print('Done reading')

    writer = NNUEWriter(nnue)
    with open(target_fn, 'wb') as f:
        f.write(writer.buf)
    print('Done writing')
*/
}
