#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

#include "training_data_loader.h"


struct Packed {
    char bytes[32];
};

int64_t read_binpack(std::string filename) {

    int64_t total = 0;
    int64_t pos_count = 0;
    int64_t move_count = 0;

    std::ifstream inf;
    inf.open(filename, std::ios::binary);

    binpack::PackedTrainingDataEntry packed;
    while (!inf.eof()) {
        bool is_ok = (bool)inf.read(reinterpret_cast<char*>(&packed), sizeof(binpack::PackedTrainingDataEntry));

        if (!is_ok) {
            break;
        }

        pos_count++;
        total++;

        int16_t numPlies = 0;
        char plyBytes[2];
        inf.read(reinterpret_cast<char*>(plyBytes), 2);

        numPlies = plyBytes[0];
        numPlies = (numPlies << 8) | plyBytes[1];

        if (numPlies > 0) {
            std::cout << "ply = " << numPlies << std::endl;
        }

        for (int i = 0; i < numPlies; i++) {
            uint16_t move, score;
            inf.read(reinterpret_cast<char*>(&move), 2);
            inf.read(reinterpret_cast<char*>(&score), 2);
            move_count++;
            total++;
        }

        if (pos_count % 100000 == 0) {
            std::cout << "Total = " << total << std::endl;
            std::cout << "Pos count = " << pos_count << std::endl;
            std::cout << "Move count = " << move_count << std::endl;
        }
    }
    
    inf.close();

    std::cout << "Total = " << total << std::endl;
    std::cout << "Pos count = " << pos_count << std::endl;
    std::cout << "Move count = " << move_count << std::endl;
    return total;
}

void head_bin_file(std::string in_fn, std::string out_fn, int64_t limit) {

    std::ifstream inf;
    inf.open(in_fn, std::ios::binary);

    std::ofstream outf;
    outf.open(out_fn, std::ios::binary);

    int64_t total = 0;
    binpack::nodchip::PackedSfenValue sfen[10001];
    while (!inf.eof()) {
        bool is_ok = (bool)inf.read(reinterpret_cast<char*>(&(sfen[total])), sizeof(binpack::nodchip::PackedSfenValue));
        if (!is_ok) {
            break;
        }
        total++;
        if (total >= limit) {
            std::cout << "Total = " << total << std::endl;
            break;
        }
    }
    
    inf.close();


    for (int i = 0; i < limit; i++) {
        outf.write(reinterpret_cast<char*>(&(sfen[limit - 1 - i])), sizeof(binpack::nodchip::PackedSfenValue));
    }
    outf.close();
}


int main() {
    //head_bin_file("/media/mc/Fastdata/Stockfish-NNUE/validate1m/val_1m_d14.bin", "/media/mc/Fastdata/Stockfish-NNUE/validate1m/val_1m_d14_top10k.bin", 10000);
    //binpack::convertBinToBinpack("/media/mc/Fastdata/Stockfish-NNUE/validate1m/val_1m_d14_top10k.bin", "./val_1m_d14_10k_compressed.binpack", std::ios::app | std::ios::binary, false);
    //binpack::convertBinToPlain("/media/mc/Fastdata/Stockfish-NNUE/validate1m/val_1m_d14_top10k.bin", "./val_1m_d14_10k_compressed.txt", std::ios::app | std::ios::binary, false);
    //binpack::convertBinpackToPlain("/media/mc/Fastdata/Stockfish-NNUE/validate1m/val_1m_d14_10k_compressed.binpack", "./val_1m_d14_10k_compressed.txt", std::ios::app | std::ios::binary, false);
    read_binpack("/media/mc/Fastdata/Stockfish-NNUE/validate1m/val_1m_d14_10k_compressed.binpack");
    return 0;
}