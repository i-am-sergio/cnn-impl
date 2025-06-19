#pragma once
#include "Tensor.hpp"
#include "Math.hpp"
#include "Layer.hpp"
#include "Optimizer.hpp"

#include <random>
#include <algorithm>
#include <cmath>
#include <unordered_map> 
#include <memory>        

// Capa Dropout: Apaga neuronas aleatoriamente durante el entrenamiento
class Dropout : public Layer {
private:
    float rate;       // Porcentaje de neuronas que se apagan
    Tensor mask;      // Mascara binaria que indica que neuronas se apagan
    bool is_training; // Indica si esta en modo entrenamiento o inferencia

    // Generador aleatorio para aplicar dropout
    std::default_random_engine rng;
    std::uniform_real_distribution<float> dist;

public:
    // Constructor: Inicializa tasa de dropout y distribucion aleatoria
    Dropout(float rate_) : rate(rate_), is_training(true), dist(0.0f, 1.0f) {
        if (rate_ < 0.0f || rate_ >= 1.0f) {
            throw std::invalid_argument("Dropout rate must be between 0.0 and 1.0 (exclusive at 1.0).");
        }
    }

    // Cambia el modo entre entrenamiento e inferencia
    void set_training_mode(bool training) {
        is_training = training;
    }

    // Dropout no necesita gradientes propios
    void zero_grad() override {}

    // Propagacion hacia adelante
    Tensor forward(const Tensor& input) override {
        Tensor output = input; // Copia de entrada

        if (is_training) {
            mask = Tensor(input.shape); // Mascara del mismo tamaño
            auto& mask_data = mask.data;
            auto& output_data = output.data;
            float scale = 1.0f / (1.0f - rate); // Escalado por dropout

            // Aplicar dropout: si pasa el umbral, mantener la neurona
            #pragma omp parallel for
            for (size_t i = 0; i < input.get_size(); ++i) {
                if (dist(rng) > rate) {
                    mask_data[i] = 1.0f;
                    output_data[i] *= scale;
                } else {
                    mask_data[i] = 0.0f;
                    output_data[i] = 0.0f;
                }
            }
        }
        return output; // En inferencia no se modifica la entrada
    }

    // Propagacion hacia atras
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input = grad_output; // Copia del gradiente de salida

        if (is_training) {
            const auto& mask_data = mask.data;
            auto& grad_input_data = grad_input.data;
            float scale = 1.0f / (1.0f - rate);

            // Aplicar la misma mascara y escalado al gradiente
            #pragma omp parallel for
            for (size_t i = 0; i < grad_output.get_size(); ++i) {
                grad_input_data[i] *= mask_data[i] * scale;
            }
        }

        return grad_input;
    }

    // Dropout no tiene parametros que actualizar
    void update_parameters(Optimizer& optimizer) override {}

    // Guardar capa en archivo
    void save(std::ostream& out) const {
        out << "Dropout " << rate << "\n";
    }

    // Cargar capa desde archivo
    void load(std::istream& in) {
        std::string type_str;
        in >> type_str;
        if (type_str != "Dropout") {
            throw std::runtime_error("Error al cargar la capa: Se esperaba 'Dropout'");
        }
        in >> rate;
        dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
        is_training = true;
    }
};