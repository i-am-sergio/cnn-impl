#include "Tensor.hpp"
#include "Layer.hpp"
#include "Dropout.hpp"
#include "NeuralNetwork.hpp"
#include "Reader.hpp"
#include "Utils.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <iomanip>

// Hiperparametros de Entrenamiento
const int EPOCHS = 2;
const float LEARNING_RATE = 0.001f;
const string LOSS_FUNCTION = "cross-entropy";
const string OPTIMIZER = "sgd"; // sgd / rmsprop / adam
const int BATCH_SIZE = 10;

void test_model(NeuralNetwork &model, vector<Tensor> &X_test, vector<Tensor> &Y_test);

int main()
{
    NeuralNetwork model;

    // Arquitectura
    model.add_layer(dense(784, 72, "relu", 0.001));
    model.add_layer(dense(72, 48, "relu", 0.001));
    model.add_layer(dense(48, 10, "softmax", 0.001));

    model.compile(LOSS_FUNCTION, OPTIMIZER, LEARNING_RATE);
    cout << "Modelo compilado exitosamente." << endl;

    // Cargar datos
    vector<vector<float>> raw_X_train, raw_Y_train;
    Reader::load_bin("./database/mnist_train.bin", raw_X_train, raw_Y_train, 60000);
    cout << "Nro datos de entrenamiento cargados: " << raw_X_train.size() << endl;

    vector<vector<float>> raw_X_test, raw_Y_test;
    Reader::load_bin("./database/mnist_test.bin", raw_X_test, raw_Y_test, 10000);

    vector<Tensor> X_test = to_tensor_batch_1D(raw_X_test);
    vector<Tensor> Y_test = to_tensor_batch_1D(raw_Y_test);

    // Dividir en train y validation (80% train, 20% validation) ------
    size_t split_idx = raw_X_train.size() * 0.8;
    vector<vector<float>> raw_X_valid(raw_X_train.begin() + split_idx, raw_X_train.end());
    vector<vector<float>> raw_Y_valid(raw_Y_train.begin() + split_idx, raw_Y_train.end());
    raw_X_train.resize(split_idx);
    raw_Y_train.resize(split_idx);

    cout << "Datos de entrenamiento: " << raw_X_train.size() << endl;
    cout << "Datos de validación: " << raw_X_valid.size() << endl;

    vector<Tensor> X_train = to_tensor_batch_1D(raw_X_train);
    vector<Tensor> Y_train = to_tensor_batch_1D(raw_Y_train);
    vector<Tensor> X_valid = to_tensor_batch_1D(raw_X_valid);
    vector<Tensor> Y_valid = to_tensor_batch_1D(raw_Y_valid);
    // ----------
    // vector<Tensor> X_train = to_tensor_batch(raw_X_train);
    // vector<Tensor> Y_train = to_tensor_batch(raw_Y_train);

    auto start = start_timer();
    model.fit(X_train, Y_train, X_valid, Y_valid, EPOCHS, BATCH_SIZE, 1, true);
    double duration = stop_timer(start);
    print_duration(duration, "Tiempo de entrenamiento");

    test_model(model, X_test, Y_test);

    // model.save_model("784x72x48x10_adam_50ep_2bz.txt");

    return 0;
}

// Probar model con Test set
void test_model(NeuralNetwork &model, vector<Tensor> &X_test, vector<Tensor> &Y_test)
{
    int correct = 0;
    int total = X_test.size();

    for (int i = 0; i < total; i++)
    {
        Tensor pred = model.predict(X_test[i]);
        int pred_label = argmax(pred);
        int true_label = argmax(Y_test[i]);

        if (pred_label == true_label)
        {
            correct++;
        }
    }

    float accuracy = 100.0f * correct / total;
    cout << "Precisión en test: " << fixed << setprecision(2) << accuracy << "%" << endl;
}