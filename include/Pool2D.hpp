#pragma once
#include "Tensor.hpp"
#include "Layer.hpp"
#include <algorithm>
#include <limits>
#include <string>

enum class PoolingType
{
    MAX,
    MIN,
    AVERAGE
};

class Pooling2D : public Layer
{
public:
    size_t pool_size;
    size_t stride;
    PoolingType type;
    Tensor last_input;

    Pooling2D(size_t pool_size = 2, size_t stride = 2, PoolingType type = PoolingType::MAX)
        : pool_size(pool_size), stride(stride), type(type) {}

    Tensor forward(const Tensor &input) override
    {
        last_input = input;

        size_t batch = input.shape[0];
        size_t channels = input.shape[1];
        size_t in_height = input.shape[2];
        size_t in_width = input.shape[3];

        size_t out_height = (in_height - pool_size) / stride + 1;
        size_t out_width = (in_width - pool_size) / stride + 1;

        Tensor output({batch, channels, out_height, out_width});

        for (size_t b = 0; b < batch; ++b)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                for (size_t oh = 0; oh < out_height; ++oh)
                {
                    for (size_t ow = 0; ow < out_width; ++ow)
                    {
                        float result;
                        if (type == PoolingType::MAX)
                            result = -std::numeric_limits<float>::infinity();
                        else if (type == PoolingType::MIN)
                            result = std::numeric_limits<float>::infinity();
                        else
                            result = 0.0f;

                        for (size_t ph = 0; ph < pool_size; ++ph)
                        {
                            for (size_t pw = 0; pw < pool_size; ++pw)
                            {
                                size_t ih = oh * stride + ph;
                                size_t iw = ow * stride + pw;

                                if (ih >= in_height || iw >= in_width)
                                    continue;
                                size_t idx = b * channels * in_height * in_width +
                                             c * in_height * in_width +
                                             ih * in_width + iw;

                                float val = input.data[idx];

                                if (type == PoolingType::MAX)
                                    result = std::max(result, val);
                                else if (type == PoolingType::MIN)
                                    result = std::min(result, val);
                                else
                                    result += val;
                            }
                        }

                        if (type == PoolingType::AVERAGE)
                            result /= (pool_size * pool_size);

                        size_t out_idx = b * channels * out_height * out_width +
                                         c * out_height * out_width +
                                         oh * out_width + ow;

                        output.data[out_idx] = result;
                    }
                }
            }
        }

        return output;
    }

    Tensor backward(const Tensor &grad_output) override
    {
        Tensor grad_input(last_input.shape);
        grad_input.fill(0.0f);

        size_t batch = last_input.shape[0];
        size_t channels = last_input.shape[1];
        size_t in_height = last_input.shape[2];
        size_t in_width = last_input.shape[3];
        size_t out_height = grad_output.shape[2];
        size_t out_width = grad_output.shape[3];

        for (size_t b = 0; b < batch; ++b)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                for (size_t oh = 0; oh < out_height; ++oh)
                {
                    for (size_t ow = 0; ow < out_width; ++ow)
                    {
                        size_t out_idx = b * channels * out_height * out_width +
                                         c * out_height * out_width +
                                         oh * out_width + ow;

                        float grad = grad_output.data[out_idx];

                        size_t max_ih = 0, max_iw = 0;
                        float best_val = (type == PoolingType::MIN)
                                             ? std::numeric_limits<float>::infinity()
                                             : -std::numeric_limits<float>::infinity();

                        for (size_t ph = 0; ph < pool_size; ++ph)
                        {
                            for (size_t pw = 0; pw < pool_size; ++pw)
                            {
                                size_t ih = oh * stride + ph;
                                size_t iw = ow * stride + pw;

                                if (ih >= in_height || iw >= in_width)
                                    continue;
                                size_t in_idx = b * channels * in_height * in_width +
                                                c * in_height * in_width +
                                                ih * in_width + iw;

                                float val = last_input.data[in_idx];

                                if (type == PoolingType::MAX && val > best_val)
                                {
                                    best_val = val;
                                    max_ih = ih;
                                    max_iw = iw;
                                }
                                else if (type == PoolingType::MIN && val < best_val)
                                {
                                    best_val = val;
                                    max_ih = ih;
                                    max_iw = iw;
                                }
                            }
                        }

                        for (size_t ph = 0; ph < pool_size; ++ph)
                        {
                            for (size_t pw = 0; pw < pool_size; ++pw)
                            {
                                size_t ih = oh * stride + ph;
                                size_t iw = ow * stride + pw;
                                size_t in_idx = b * channels * in_height * in_width +
                                                c * in_height * in_width +
                                                ih * in_width + iw;

                                if (type == PoolingType::AVERAGE)
                                {
                                    grad_input.data[in_idx] += grad / (pool_size * pool_size);
                                }
                                else if (ih == max_ih && iw == max_iw)
                                {
                                    grad_input.data[in_idx] += grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        return grad_input;
    }

    void update_parameters(Optimizer &) override {}
    void zero_grad() override {}
};
