/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/GluFfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T>
struct GlmDecoderLayerWeight {
public:
    GlmDecoderLayerWeight() = default;
    GlmDecoderLayerWeight(const int hidden_units,
                           const int inter_size,
                           const int tensor_para_size = 1,
                           const int tensor_para_rank = 0);
    ~GlmDecoderLayerWeight();
    GlmDecoderLayerWeight(const GlmDecoderLayerWeight& other);
    GlmDecoderLayerWeight& operator=(const GlmDecoderLayerWeight& other);

    void loadModel(std::string dir_path, FtCudaDataType model_file_type);

    AttentionWeight<T> self_attention_weights;
    LayerNormWeight<T> self_attn_layernorm_weights;
    GluFfnWeight<T> glu_ffn_weights;
    LayerNormWeight<T> glu_ffn_layernorm_weights;


private:
    int hidden_units_;
    int inter_size_;
    int tensor_para_size_;
    int tensor_para_rank_;
    bool is_maintain_buffer = false;
    T* weights_ptr[14];

    void setWeightPtr();
    void mallocWeights();
};

}  // namespace fastertransformer
