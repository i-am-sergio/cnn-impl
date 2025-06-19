#pragma once
#include "Tensor.hpp"
#include "Layer.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>


// Capa de convolución 2D para redes neuronales
class Conv2D : public Layer {
public:
    size_t input_channels;  // Canales de entrada (ej: 3 para RGB)
    size_t output_channels; // Número de filtros/canales de salida
    size_t kernel_size;     // Tamaño del kernel (cuadrado)
    size_t stride;         // Paso de la convolución
    size_t padding;        // Relleno en los bordes
    
    Tensor kernels;        // Filtros/kernels [output_channels, input_channels, kernel_size, kernel_size]
    Tensor bias;           // Sesgos [output_channels]
    
    // Cache para backpropagation
    Tensor last_input;     // Última entrada [batch, in_channels, height, width]
    Tensor grad_kernels;   // Gradiente de los kernels
    Tensor grad_bias;      // Gradiente de los sesgos

    // Constructor
    Conv2D(size_t in_channels, size_t out_channels, 
          size_t kernel_size = 3, size_t stride = 1, 
          size_t padding = 0)
        : input_channels(in_channels), output_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding) {
        
        // Inicializar kernels y bias
        kernels = Tensor({out_channels, in_channels, kernel_size, kernel_size});
        bias = Tensor({out_channels});
        grad_kernels = Tensor(kernels.shape);
        grad_bias = Tensor(bias.shape);
        
        initialize_parameters();
    }

    // Inicialización de parámetros (He initialization)
    void initialize_parameters() {
        float stddev = sqrt(2.0f / (input_channels * kernel_size * kernel_size));
        std::default_random_engine generator;
        std::normal_distribution<float> dist(0.0f, stddev);
        
        for (float& val : kernels.data) {
            val = dist(generator);
        }
        
        for (float& val : bias.data) {
            val = 0.01f * dist(generator);
        }
    }

    // Forward pass
    Tensor forward(const Tensor& input) override {
        last_input = input;
        
        // Dimensiones de entrada [batch, in_channels, height, width]
        size_t batch_size = input.shape[0];
        size_t in_height = input.shape[2];
        size_t in_width = input.shape[3];
        
        // Calcular dimensiones de salida
        size_t out_height = (in_height + 2*padding - kernel_size) / stride + 1;
        size_t out_width = (in_width + 2*padding - kernel_size) / stride + 1;
        
        Tensor output({batch_size, output_channels, out_height, out_width});
        
        // Aplicar convolución para cada elemento del batch
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t oc = 0; oc < output_channels; ++oc) {
                for (size_t oh = 0; oh < out_height; ++oh) {
                    for (size_t ow = 0; ow < out_width; ++ow) {
                        
                        float sum = bias.data[oc];
                        
                        // Aplicar kernel
                        for (size_t ic = 0; ic < input_channels; ++ic) {
                            for (size_t kh = 0; kh < kernel_size; ++kh) {
                                for (size_t kw = 0; kw < kernel_size; ++kw) {
                                    
                                    size_t ih = oh*stride + kh - padding;
                                    size_t iw = ow*stride + kw - padding;
                                    
                                    // Comprobar bordes
                                    if (ih < in_height && iw < in_width) {
                                        size_t input_idx = b * input.shape[1] * input.shape[2] * input.shape[3] +
                                                         ic * input.shape[2] * input.shape[3] +
                                                         ih * input.shape[3] + 
                                                         iw;
                                        
                                        size_t kernel_idx = oc * kernels.shape[1] * kernels.shape[2] * kernels.shape[3] +
                                                           ic * kernels.shape[2] * kernels.shape[3] +
                                                           kh * kernels.shape[3] + 
                                                           kw;
                                        
                                        sum += input.data[input_idx] * kernels.data[kernel_idx];
                                    }
                                }
                            }
                        }
                        
                        // Guardar resultado
                        size_t output_idx = b * output.shape[1] * output.shape[2] * output.shape[3] +
                                          oc * output.shape[2] * output.shape[3] +
                                          oh * output.shape[3] + 
                                          ow;
                        
                        output.data[output_idx] = sum;
                    }
                }
            }
        }
        
        return output;
    }

    // Backward pass
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(last_input.shape);
        
        // Dimensiones
        size_t batch_size = last_input.shape[0];
        size_t in_height = last_input.shape[2];
        size_t in_width = last_input.shape[3];
        size_t out_height = grad_output.shape[2];
        size_t out_width = grad_output.shape[3];
        
        // Inicializar gradientes a cero
        grad_kernels.fill(0.0f);
        grad_bias.fill(0.0f);
        
        // Calcular gradientes
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t oc = 0; oc < output_channels; ++oc) {
                for (size_t oh = 0; oh < out_height; ++oh) {
                    for (size_t ow = 0; ow < out_width; ++ow) {
                        
                        float grad = grad_output.data[
                            b * grad_output.shape[1] * grad_output.shape[2] * grad_output.shape[3] +
                            oc * grad_output.shape[2] * grad_output.shape[3] +
                            oh * grad_output.shape[3] + 
                            ow
                        ];
                        
                        // Gradiente del bias
                        grad_bias.data[oc] += grad;
                        
                        for (size_t ic = 0; ic < input_channels; ++ic) {
                            for (size_t kh = 0; kh < kernel_size; ++kh) {
                                for (size_t kw = 0; kw < kernel_size; ++kw) {
                                    
                                    size_t ih = oh*stride + kh - padding;
                                    size_t iw = ow*stride + kw - padding;
                                    
                                    if (ih < in_height && iw < in_width) {
                                        // Posiciones en tensores
                                        size_t input_idx = b * last_input.shape[1] * last_input.shape[2] * last_input.shape[3] +
                                                         ic * last_input.shape[2] * last_input.shape[3] +
                                                         ih * last_input.shape[3] + 
                                                         iw;
                                        
                                        size_t kernel_idx = oc * kernels.shape[1] * kernels.shape[2] * kernels.shape[3] +
                                                          ic * kernels.shape[2] * kernels.shape[3] +
                                                          kh * kernels.shape[3] + 
                                                          kw;
                                        
                                        // Gradiente de los kernels
                                        grad_kernels.data[kernel_idx] += last_input.data[input_idx] * grad;
                                        
                                        // Gradiente de la entrada (si es necesario)
                                        grad_input.data[input_idx] += kernels.data[kernel_idx] * grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return grad_input;
    }

    // Actualización de parámetros
    void update_parameters(Optimizer& optimizer) override {
        optimizer.update(kernels.data, grad_kernels.data);
        optimizer.update(bias.data, grad_bias.data);
    }

    // Reiniciar gradientes
    void zero_grad() override {
        grad_kernels.fill(0.0f);
        grad_bias.fill(0.0f);
    }
};