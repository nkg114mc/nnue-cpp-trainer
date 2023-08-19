#pragma once
#ifndef SERIALIZE_H
#define SERIALIZE_H

#include <string>

#include "model.h"
#include "feature_set.h"

NNUEModel load_model_from_nnuebin(std::string source_path,
                                  FeatureSetPy *feature_set);

void main_simple();

#endif // #define SERIALIZE_H