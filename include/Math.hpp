#pragma once
#include "Tensor.hpp"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <omp.h>

// Realiza el producto punto entre dos tensores:
// - 'a': Tensor de entrada (1D)
// - 'b': Tensor de pesos (2D, forma [N, M])
// Retorna un nuevo tensor con el resultado (1D, tamaño M)
Tensor dot_product(const Tensor &a, const Tensor &b) {
    const auto& a_data = a.data;      // Acceso directo a los datos
    const auto& b_data = b.data;
    const auto& b_shape = b.shape;
    
    // Dimensiones del producto punto
    size_t N = b_shape[0];  // dim. entrada
    size_t M = b_shape[1];  // dim. salida

    Tensor result({M});     // Tensor resultado
    
    // Paraleliza el cálculo con OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < M; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < N; j++) {
            // Formula del producto punto
            sum += a_data[j] * b_data[j * M + i]; 
        }
        result.data[i] = sum;
    }
    
    return result;
}


// Devuelve el índice del elemento con mayor valor en el tensor
inline int argmax(const Tensor& tensor) {
    if (tensor.data.empty()) {
        throw std::runtime_error("Tensor vacío en argmax");
    }
    
    const auto& data = tensor.data;
    int max_index = 0;
    float max_val = data[0];
    
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_index = static_cast<int>(i);
        }
    }
    
    return max_index;
}