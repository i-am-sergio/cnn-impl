#include <iostream>
#include "Tensor.hpp" 
#include "Utils.hpp"

using namespace std;

int main() {
    // Crear tensor 2D de tamaño 20x12 (240 elementos)
    Tensor tensor2D({20, 12});
    for (size_t i = 0; i < tensor2D.get_size(); ++i) {
        tensor2D.data[i] = static_cast<float>(i);
    }

    // Crear tensor 3D de tamaño 10x4x6 (240 elementos)
    Tensor tensor3D({10, 4, 6});
    for (size_t i = 0; i < tensor3D.get_size(); ++i) {
        tensor3D.data[i] = static_cast<float>(i) / 10.0f; 
    }

    cout << "Tensor 2D (20x12):" << endl;
    cout << tensor2D << endl;
    cout << tensor2D.shape << endl;

    cout << "\nTensor 3D (10x4x6):" << endl;
    cout << tensor3D << endl;

    return 0;
}