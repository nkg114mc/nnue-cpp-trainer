#pragma once
#ifndef FEATURE_SET_PY_H
#define FEATURE_SET_PY_H

#include <cstdint>


class FeatureSetPy {
public:
    FeatureSetPy();
    ~FeatureSetPy();

    void get_virtual_feature_ranges();
    void get_virtual_to_real_features_gather_indices();

    inline uint32_t get_hash() { return hash; }
    inline std::string get_name()  { return name; }
    inline int get_num_real_features() { return num_real_features; }
    inline int get_num_virtual_features()  { return num_virtual_features; }
    inline int get_num_features()  { return num_features; }

private:

    uint32_t hash;
    std::string name;
    int num_real_features;
    int num_virtual_features; // = sum(feature.num_virtual_features for feature in features)
    int num_features;
};

#endif // #define FEATURE_SET_PY_H