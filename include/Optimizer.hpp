#pragma once

#include "Tensor.hpp"

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <cmath>

using namespace std;

class Optimizer {
protected:
    float learning_rate;
    // Mapas para almacenar estados especificos de cada optimizador 
    unordered_map<const float*, Tensor> m_moments; // Para Adam (1er momento)
    unordered_map<const float*, Tensor> v_moments; // Para Adam (2do momento) / RMSProp (promedio de gradientes cuadrados)
    unordered_map<const float*, int> t_steps;      // Para Adam (pasos temporales para correccion de sesgo)

public:
    Optimizer(float lr) : learning_rate(lr) {}
    virtual ~Optimizer() = default;

    // Metodo para actualizar los parametros de un tensor
    virtual void update(vector<float>& param_data, const vector<float>& grad_data) = 0;
};


class SGD_Optimizer : public Optimizer {
public:
    SGD_Optimizer(float lr) : Optimizer(lr) {}

    void update(vector<float>& param_data, const vector<float>& grad_data) override {
        for (size_t i = 0; i < param_data.size(); ++i) {
            param_data[i] -= learning_rate * grad_data[i];
        }
    }
};


class RMSProp_Optimizer : public Optimizer {
private:
    float beta = 0.9f;      // Factor de decaimiento exponencial
    float epsilon = 1e-8f;  // Constante para evitar divisiones por cero

public:
    RMSProp_Optimizer(float lr, float beta_ = 0.9f) : Optimizer(lr), beta(beta_) {}

    void update(vector<float>& param_data, const vector<float>& grad_data) override {
        const float* param_key = param_data.data(); // Identificador unico del tensor de parametros

        // Intentamos obtener el estado previo del acumulador v
        auto v_it = v_moments.find(param_key);
        if (v_it == v_moments.end()) {
            // Si no existe, lo inicializamos en cero
            v_it = v_moments.emplace(param_key, Tensor({param_data.size()})).first;
            v_it->second.fill(0.0f);
        }

        auto& v_data = v_it->second.data;     // Referencia directa a los datos de v
        const size_t size = param_data.size();
        const float one_minus_beta = 1.0f - beta;   // Precomputamos para eficiencia

        for (size_t i = 0; i < size; ++i) {
            float g = grad_data[i];                // Gradiente actual
            float& v = v_data[i];                  // Valor acumulado de v
            v = beta * v + one_minus_beta * g * g; // Actualizacion de v con promedio exponencial
            param_data[i] -= learning_rate * g / (std::sqrt(v) + epsilon); // Actualizacion del parametro
        }
    }
};

class Adam_Optimizer : public Optimizer {
private:
    float beta1;
    float beta2;
    float epsilon;
    bool use_weight_decay;
    float lambda;

public:
    Adam_Optimizer(float lr, float beta1_ = 0.8f, float beta2_ = 0.9f, 
                  float eps = 1e-6f, float lambda_ = 0.0f) 
        : Optimizer(lr), beta1(beta1_), beta2(beta2_), 
          epsilon(eps), lambda(lambda_) {}

    void update(vector<float>& param_data, const vector<float>& grad_data) override {
        const float* param_key = param_data.data();
        
        // Inicialización de momentos
        if (m_moments[param_key].data.empty()) {
            m_moments[param_key] = Tensor({param_data.size()});
            v_moments[param_key] = Tensor({param_data.size()});
            m_moments[param_key].fill(0.0f);
            v_moments[param_key].fill(0.0f);
            t_steps[param_key] = 0;
        }

        auto& m = m_moments[param_key].data;
        auto& v = v_moments[param_key].data;
        int& t = t_steps[param_key];
        t++;

        for (size_t i = 0; i < param_data.size(); ++i) {
            float g = grad_data[i];
            
            // Weight decay
            if (lambda > 0.0f) {
                g += 2 * lambda * param_data[i];
            }

            // Actualización de momentos
            m[i] = beta1 * m[i] + (1 - beta1) * g;
            v[i] = beta2 * v[i] + (1 - beta2) * g * g;

            // Corrección de sesgo
            float m_hat = m[i] / (1 - pow(beta1, t));
            float v_hat = v[i] / (1 - pow(beta2, t));

            // Actualización de parámetros
            param_data[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
    }
};