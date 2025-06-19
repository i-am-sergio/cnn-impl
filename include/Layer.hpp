#pragma once

#include "Tensor.hpp"
#include "Optimizer.hpp"

// Clase base abstracta para todas las capas de una red neuronal
class Layer {
public:
    virtual ~Layer() = default;

    // Procesa la entrada y devuelve la salida de la capa
    virtual Tensor forward(const Tensor& input) = 0;

    // Calcula los gradientes respecto a la entrada y pesos
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // Actualiza los parámetros usando el optimizador
    virtual void update_parameters(Optimizer& optimizer) = 0;

    // Reinicia los gradientes acumulados a cero
    virtual void zero_grad() = 0;
};