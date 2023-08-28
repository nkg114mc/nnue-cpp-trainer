#include <torch/torch.h>
#include <cstdint>
#include <fstream>

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
        # InputSlice hash
        prev_hash = 0xEC42E90D
        prev_hash ^= (M.L1 * 2)

        # Fully connected layers
        layers = [model.l1, model.l2, model.output]
        for layer in layers:
            layer_hash = 0xCC03DAE4
            layer_hash += layer.out_features
            layer_hash ^= prev_hash >> 1
            layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
            if layer.out_features != 1:
                # Clipped ReLU hash
                layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
            prev_hash = layer_hash
        return layer_hash*/
        return 0;
    }

/*
    NNUEWriter(NNUEModel &model) {
        self.buf = bytearray()

        fc_hash = self.fc_hash(model)
        self.write_header(model, fc_hash)
        self.int32(model.feature_set.hash ^ (M.L1 * 2))  // Feature transformer hash
        self.write_feature_transformer(model)
        self.int32(fc_hash)  // FC layers hash
        self.write_fc_layer(model.l1, false);
        self.write_fc_layer(model.l2, false);
        self.write_fc_layer(model.output, true);
    }

    void write_header(NNUEModel &model, uint32_t fc_hash) {
        self.int32(VERSION)  # version
        self.int32(fc_hash ^ model.feature_set.hash ^ (M.L1 * 2))  # halfkp network hash
        # description = b"Features=HalfKP(Friend)[41024->256x2],"
        # description += b"Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32]"
        # description += b"(ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
        description = model.description.encode('utf-8')
        self.int32(len(description))  # Network definition
        self.buf.extend(description)
        print("description:", model.description)
    }

    void coalesce_ft_weights(NNUEModel &model, layer) {
        weight = layer.weight.data
        indices = model.feature_set.get_virtual_to_real_features_gather_indices()
        weight_coalesced = weight.new_zeros((model.feature_set.num_real_features, weight.shape[1]))
        for i_real, is_virtual in enumerate(indices):
            weight_coalesced[i_real, :] = sum(weight[i_virtual, :] for i_virtual in is_virtual)

        return weight_coalesced
    }

    void write_feature_transformer(NNUEModel &model) {
        # int16 bias = round(x * 127)
        # int16 weight = round(x * 127)
        layer = model.input
        bias = layer.bias.data
        bias = bias.mul(127).round().to(torch.int16)
        ascii_hist('ft bias:', bias.numpy())
        self.buf.extend(bias.flatten().numpy().tobytes())

        weight = self.coalesce_ft_weights(model, layer)
        weight = weight.mul(127).round().to(torch.int16)
        ascii_hist('ft weight:', weight.numpy())
        # weights stored as [41024][256]
        self.buf.extend(weight.flatten().numpy().tobytes())
    }

    void write_fc_layer( layer, bool is_output) {
        # FC layers are stored as int8 weights, and int32 biases
        kWeightScaleBits = 6
        kActivationScale = 127.0
        if not is_output:
            kBiasScale = (1 << kWeightScaleBits) * kActivationScale  # = 8128
        else:
            kBiasScale = 9600.0  # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
        kWeightScale = kBiasScale / kActivationScale  # = 64.0 for normal layers
        kMaxWeight = 127.0 / kWeightScale  # roughly 2.0

        # int32 bias = round(x * kBiasScale)
        # int8 weight = round(x * kWeightScale)
        bias = layer.bias.data
        bias = bias.mul(kBiasScale).round().to(torch.int32)
        ascii_hist('fc bias:', bias.numpy())
        self.buf.extend(bias.flatten().numpy().tobytes())
        weight = layer.weight.data
        clipped = torch.count_nonzero(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
        total_elements = torch.numel(weight)
        clipped_max = torch.max(torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight))
        print("layer has {}/{} clipped weights. Exceeding by {} the maximum {}.".format(clipped, total_elements,
                                                                                        clipped_max, kMaxWeight))
        weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
        ascii_hist('fc weight:', weight.numpy())
        # FC inputs are padded to 32 elements for simd.
        num_input = weight.shape[1]
        if num_input % 32 != 0:
            num_input += 32 - (num_input % 32)
            new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
            new_w[:, :weight.shape[1]] = weight
            weight = new_w
        # Stored as [outputs][inputs], so we can flatten
        self.buf.extend(weight.flatten().numpy().tobytes())
    }

    void int32(self, v) {
        self.buf.extend(struct.pack("<I", v))
    }

*/
private:

/*
// read evaluation function parameters
template <typename T>
bool ReadParameters(std::istream& stream, const AlignedPtr<T>& pointer) {
  std::uint32_t header;
  stream.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!stream || header != T::GetHashValue()) return false;
  return pointer->ReadParameters(stream);
}

// write evaluation function parameters
template <typename T>
bool WriteParameters(std::ostream& stream, const AlignedPtr<T>& pointer) {
  constexpr std::uint32_t header = T::GetHashValue();
  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  return pointer->WriteParameters(stream);
}

}  // namespace Detail

// Initialize the evaluation function parameters
void Initialize() {
  Detail::Initialize(feature_transformer);
  Detail::Initialize(network);
}

}  // namespace

// read the header
bool ReadHeader(std::istream& stream,
  std::uint32_t* hash_value, std::string* architecture) {
  std::uint32_t version, size;
  stream.read(reinterpret_cast<char*>(&version), sizeof(version));
  stream.read(reinterpret_cast<char*>(hash_value), sizeof(*hash_value));
  stream.read(reinterpret_cast<char*>(&size), sizeof(size));
  if (!stream || version != kVersion) return false;
  architecture->resize(size);
  stream.read(&(*architecture)[0], size);
  return !stream.fail();
}

// write the header
bool WriteHeader(std::ostream& stream,
  std::uint32_t hash_value, const std::string& architecture) {
  stream.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
  stream.write(reinterpret_cast<const char*>(&hash_value), sizeof(hash_value));
  const std::uint32_t size = static_cast<std::uint32_t>(architecture.size());
  stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
  stream.write(architecture.data(), size);
  return !stream.fail();
}

// read evaluation function parameters
bool ReadParameters(std::istream& stream) {
  std::uint32_t hash_value;
  std::string architecture;
  if (!ReadHeader(stream, &hash_value, &architecture)) return false;
  if (hash_value != kHashValue) return false;
  if (!Detail::ReadParameters(stream, feature_transformer)) return false;
  if (!Detail::ReadParameters(stream, network)) return false;
  return stream && stream.peek() == std::ios::traits_type::eof();
}

// write evaluation function parameters
bool WriteParameters(std::ostream& stream) {
  if (!WriteHeader(stream, kHashValue, GetArchitectureString())) return false;
  if (!Detail::WriteParameters(stream, feature_transformer)) return false;
  if (!Detail::WriteParameters(stream, network)) return false;
  return !stream.fail();
}
*/

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
        return length;
    }

    torch::Tensor read_tensor(const torch::Dtype &dtype, const torch::IntArrayRef &shape) {
        //d = numpy.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
        //d = torch.from_numpy(d.astype(numpy.float32))
        //d = d.reshape(shape)
        //return d
        auto options = torch::TensorOptions().dtype(dtype);
        torch::Tensor tensor = torch::full(shape, 0, options);
        uint32_t length = get_total_size(shape);
        inf.read(reinterpret_cast<char*>(tensor.data_ptr()), sizeof(dtype) * length);
        return tensor;
    }

    //void read_feature_transformer(FeatureTransformerSliceEmulate &layer) {
    void read_feature_transformer(FeatTransSlow &layer) {
        layer->bias = read_tensor(torch::kInt16, layer->bias.sizes()).divide(127.0);
        // weights stored as [41024][256]
        auto weights_int = read_tensor(torch::kInt16, layer->weight.sizes());
        layer->weight = weights_int.divide(127.0);
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
