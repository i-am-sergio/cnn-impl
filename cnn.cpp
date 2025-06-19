#include "Tensor.hpp"
#include "Layer.hpp"
#include "Conv2D.hpp"
#include "Flatten.hpp"
#include "Dropout.hpp"
#include "NeuralNetwork.hpp"
#include "Reader.hpp"
#include "Utils.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>


// Hiperparámetros
const int EPOCHS = 5;
const float LEARNING_RATE = 0.001f;
const string LOSS_FUNCTION = "cross-entropy";
const string OPTIMIZER = "sgd";
const int BATCH_SIZE = 10;

void test_model(NeuralNetwork& model, vector<Tensor>& X_test, vector<Tensor>& Y_test);

int main() {
    NeuralNetwork model;
    
    // Arquitectura CNN corregida
    model.add_layer(conv2d(1, 32));  // 1 canal de entrada (MNIST), 32 filtros
    model.add_layer(flatten());      // Aplanar para conectar a densa
    model.add_layer(dense(32*28*28, 256, "relu"));
    model.add_layer(dropout(0.4f));
    model.add_layer(dense(256, 10, "softmax"));

    model.compile(LOSS_FUNCTION, OPTIMIZER, LEARNING_RATE);
    cout << "Modelo CNN compilado exitosamente." << endl;

    // Cargar datos (asegurarse que Reader::load_bin carga MNIST correctamente)
    vector<vector<float>> raw_X_train, raw_Y_train;
    Reader::load_bin("../topicos-inteligencia-artificial/datasets/MNISTdataset/mnist_train.bin", raw_X_train, raw_Y_train, 60000);
    
    // Convertir a tensores 4D
    vector<Tensor> X_train = to_tensor_batch(raw_X_train);
    vector<Tensor> Y_train = to_tensor_batch(raw_Y_train);  // Esto sigue siendo 1D

    // Validación y test
    vector<vector<float>> raw_X_test, raw_Y_test;
    Reader::load_bin("../topicos-inteligencia-artificial/datasets/MNISTdataset/mnist_test.bin", raw_X_test, raw_Y_test, 10000);
    vector<Tensor> X_test = to_tensor_batch(raw_X_test);
    vector<Tensor> Y_test = to_tensor_batch(raw_Y_test);

    // Entrenamiento
    auto start = start_timer();
    model.fit(X_train, Y_train, X_test, Y_test, EPOCHS, BATCH_SIZE, 100, true);
    double duration = stop_timer(start);
    print_duration(duration, "Tiempo de entrenamiento");

    test_model(model, X_test, Y_test);
    
    return 0;
}

void test_model(NeuralNetwork& model, vector<Tensor>& X_test, vector<Tensor>& Y_test) {
    int correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < X_test.size(); i++) {
        Tensor pred = model.predict(X_test[i]);
        if (argmax(pred) == argmax(Y_test[i])) {
            correct++;
        }
    }
    cout << "Precisión en test: " << fixed << setprecision(2) 
         << (100.0f * correct / X_test.size()) << "%" << endl;
}