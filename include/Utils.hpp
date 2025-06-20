#pragma once

#include "Dense.hpp"
#include "Conv2D.hpp"
#include "Flatten.hpp"
#include "Pool2D.hpp"

#include <chrono>
#include <memory>
#include <vector>
#include <iomanip>

using namespace std;

#define RESET "\033[0m"
#define BOLD "\033[1m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"
#define MAGENTA "\033[35m"

// Cronometro
using Time = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Time>;

// Inicia el cronometro
inline TimePoint start_timer()
{
    return Time::now();
}

// Detiene el cronometro y devuelve la duracion en segundos
inline double stop_timer(const TimePoint &start_time)
{
    TimePoint end_time = Time::now();
    std::chrono::duration<double> duration = end_time - start_time;
    return duration.count();
}

// Imprime duracion en segundos con formato
inline void print_duration(double duration, const string &label)
{
    cout << label << ": " << fixed << setprecision(2) << duration << " s" << endl;
}

// Funciones auxiliares para crear capas
auto dense = [](int in, int out, const string &act, float lambda = 0.0)
{
    return std::make_unique<Dense>(in, out, act, lambda);
};

auto dropout = [](float rate)
{
    return std::make_unique<Dropout>(rate);
};

auto conv2d = [](int in_ch, int out_ch, int kernel = 3, int stride = 1, int pad = 0)
{
    return std::make_unique<Conv2D>(in_ch, out_ch, kernel, stride, pad);
};

auto flatten = []()
{
    return std::make_unique<Flatten>();
};

auto pool = [](size_t pool_size = 2, size_t stride = 2, PoolingType type = PoolingType::MAX)
{
    return std::make_unique<Pooling2D>(pool_size, stride, type);
};

// Convertir labels one-hot a tensores 1D [10] (solo para etiquetas)
vector<Tensor> to_tensor_batch_1D(const vector<vector<float>> &data)
{
    vector<Tensor> tensors;
    for (const auto &vec : data)
    {
        assert(vec.size() == 10 && "Cada etiqueta debe tener 10 valores (one-hot)");
        Tensor t({vec.size()}); // Tensor 1D [10]
        t.data = vec;
        tensors.push_back(t);
    }
    return tensors;
}

// Convertir MNIST a tensores 4D [1, 1, 28, 28] (batch=1, channels=1, height=28, width=28)
vector<Tensor> to_tensor_batch_4D(const vector<vector<float>> &data)
{
    vector<Tensor> tensors;
    for (const auto &vec : data)
    {
        assert(vec.size() == 784 && "Cada imagen MNIST debe tener 784 valores");
        Tensor t({1, 1, 28, 28}); // Formato [batch=1, canales=1, altura=28, ancho=28]
        t.data = vec;             // Los datos ya están en el orden correcto
        tensors.push_back(t);
    }
    return tensors;
}