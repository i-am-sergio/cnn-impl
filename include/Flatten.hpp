#pragma once
#include "Layer.hpp"

// Capa Flatten para aplanar tensores multidimensionales a 1D
class Flatten : public Layer {
public:
    vector<size_t> input_shape;  // Guarda la forma original para reshape en backward

    // Forward pass: aplana el input a 1D
    Tensor forward(const Tensor& input) override {
        input_shape = input.shape; // Guardar forma original
        
        // Calcular tamaño total
        size_t total_size = 1;
        for (auto dim : input_shape) {
            total_size *= dim;
        }
        
        // Crear tensor 1D
        Tensor output({total_size});
        output.data = input.data; // Compartir datos (no copiar)
        
        return output;
    }

    // Backward pass: restaura la forma original
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(input_shape);
        grad_input.data = grad_output.data; // Compartir datos (no copiar)
        return grad_input;
    }

    // No hay parámetros para actualizar
    void update_parameters(Optimizer& optimizer) override {}

    // No hay gradientes que reiniciar
    void zero_grad() override {}
};