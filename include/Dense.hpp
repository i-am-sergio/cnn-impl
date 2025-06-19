#pragma once

#include "Layer.hpp"
#include "Tensor.hpp"
#include "Optimizer.hpp"
#include "Math.hpp"

#include <random>
#include <cmath>
#include <algorithm>

// Capa densa (fully connected) para redes neuronales
class Dense : public Layer {
public:
    size_t input_dim;     // Dimension de entrada
    size_t output_dim;    // Dimension de salida
    Tensor weights;       // Matriz de pesos [input_dim x output_dim]
    Tensor bias;          // Vector de sesgos [output_dim]
    string activation;    // Tipo de funcion de activacion
    float lambda;         // Coeficiente de regularizacion L2

    // Cache para backpropagation
    Tensor last_input;    // Entrada en forward pass
    Tensor last_output;   // Salida pre-activacion
    Tensor last_activated;// Salida post-activacion
    
    // Gradientes
    Tensor grad_weights;  // dL/dW
    Tensor grad_bias;     // dL/db

    // Constructor: inicializa pesos y configura dimensiones
    Dense(size_t input_dim_, size_t output_dim_, 
          const string& activation_ = "", float lambda_ = 0.0f) 
        : input_dim(input_dim_), output_dim(output_dim_),
          activation(activation_), lambda(lambda_) {
        
        weights = Tensor({input_dim_, output_dim_});
        bias = Tensor({output_dim_});
        grad_weights = Tensor({input_dim_, output_dim_});
        grad_bias = Tensor({output_dim_});
        
        initialize_weights_random_uniform(0.1f);
    }

    // Inicializacion uniforme de pesos
    void initialize_weights_random_uniform(float range) {
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(-range, range);
        
        for (float& w : weights.data) w = dist(rng);
        for (float& b : bias.data) b = 0.01f * dist(rng);
    }

    float compute_l2_penalty() const {
        if (lambda == 0.0f) return 0.0f;
        float sum = 0.0f;
        for (float w : weights.data) {
            sum += w * w;
        }
        return lambda * sum;
    }

    void scale_gradients(float scale) {
        for (float& g : grad_weights.data) g *= scale;
        for (float& g : grad_bias.data) g *= scale;
    }

    // Reinicia gradientes a cero
    void zero_grad() override {
        grad_weights.fill(0.0f);
        grad_bias.fill(0.0f);
    }

    // Forward pass: X -> (XW + b) -> activacion
    Tensor forward(const Tensor& input) override {
        last_input = input;
        last_output = dot_product(input, weights);
        
        // Sumar bias
        for (size_t i = 0; i < output_dim; ++i) {
            last_output.data[i] += bias.data[i];
        }
        
        // Aplicar activacion
        last_activated = Tensor({output_dim});
        if (activation == "softmax") {
            last_activated = softmax(last_output);
        } else {
            for (size_t i = 0; i < output_dim; ++i) {
                last_activated.data[i] = activation_function(last_output.data[i]);
            }
        }
        
        return last_activated;
    }

    // Backward pass: calcula gradientes
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input({input_dim});
        
        if (activation == "softmax") {
            // Caso especial para softmax + cross-entropy
            for (size_t i = 0; i < output_dim; ++i) {
                float dz = grad_output.data[i];
                grad_bias.data[i] += dz;
                
                for (size_t j = 0; j < input_dim; ++j) {
                    grad_weights.data[j * output_dim + i] += dz * last_input.data[j];
                    grad_input.data[j] += dz * weights.data[j * output_dim + i];
                }
            }
        } else {
            // Caso general para otras activaciones
            for (size_t i = 0; i < output_dim; ++i) {
                float dz = grad_output.data[i] * activation_derivative(last_output.data[i]);
                grad_bias.data[i] += dz;
                
                for (size_t j = 0; j < input_dim; ++j) {
                    grad_weights.data[j * output_dim + i] += dz * last_input.data[j];
                    grad_input.data[j] += dz * weights.data[j * output_dim + i];
                }
            }
        }
        
        // Regularizacion L2
        if (lambda > 0.0f) {
            for (size_t i = 0; i < weights.data.size(); ++i) {
                grad_weights.data[i] += 2 * lambda * weights.data[i];
            }
        }
        
        return grad_input;
    }

    // Actualiza parametros usando optimizador
    void update_parameters(Optimizer& optimizer) override {
        optimizer.update(weights.data, grad_weights.data);
        optimizer.update(bias.data, grad_bias.data);
    }

private:
    // Funciones de activacion
    float activation_function(float x) const {
        if (activation == "relu") return std::max(0.0f, x);
        if (activation == "sigmoid") return 1.0f / (1.0f + exp(-x));
        if (activation == "tanh") return tanh(x);
        return x;  // Linear
    }

    // Derivadas de activacion
    float activation_derivative(float x) const {
        if (activation == "relu") return x > 0 ? 1.0f : 0.0f;
        if (activation == "sigmoid") {
            float s = 1.0f / (1.0f + exp(-x));
            return s * (1.0f - s);
        }
        if (activation == "tanh") {
            float t = tanh(x);
            return 1.0f - t * t;
        }
        return 1.0f;  // Linear
    }

    // Softmax con estabilidad numerica
    Tensor softmax(const Tensor& input) {
        Tensor result({output_dim});
        float max_val = *std::max_element(input.data.begin(), input.data.end());
        float sum_exp = 0.0f;
        
        for (size_t i = 0; i < output_dim; ++i) {
            result.data[i] = exp(input.data[i] - max_val);
            sum_exp += result.data[i];
        }
        
        for (float& val : result.data) {
            val /= sum_exp;
        }
        
        return result;
    }
};