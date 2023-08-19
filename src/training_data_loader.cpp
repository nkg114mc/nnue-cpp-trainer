
#include "training_data_loader.h"

/* benches */
#include <chrono>

int main()
{
    auto stream = create_sparse_batch_stream("HalfKP", 4, "/media/mc/Fastdata/Stockfish-NNUE/validate1m/val_1m_d14.bin", 10000, true, false, 0, false);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        SparseBatch *batch;
        batch = stream->next();
        // if (i % 100 == 0) std::cout << i << '\n';
        std::cout << "batch " << i << '\n';
        std::cout << batch->num_inputs << " " << batch->size << std::endl;
        destroy_sparse_batch(batch);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}

