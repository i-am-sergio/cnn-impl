#include "Conv2D.hpp"
#include "Dropout.hpp"
#include "Flatten.hpp"
#include "Layer.hpp"
#include "NeuralNetwork.hpp"
#include "Reader.hpp"
#include "Tensor.hpp"
#include "Utils.hpp"

#include <iostream>

// Hiperparámetros
const int EPOCHS = 10;
const float LEARNING_RATE = 0.001f;
const string LOSS_FUNCTION = "cross-entropy";
const string OPTIMIZER = "sgd";
const int BATCH_SIZE = 10;

void test_model(NeuralNetwork &model, vector<Tensor> &X_test, vector<Tensor> &Y_test);

int main() {
  NeuralNetwork model;

  // Arquitectura CNN corregida
  model.add_layer(conv2d(1, 8, 5, 2, 2));        // 8 filtros 5x5, stride=2 → [8,14,14]
  model.add_layer(pool(2, 2, PoolingType::MAX)); // [8, 7, 7]
  model.add_layer(flatten());                    // [8×7×7 = 392]
  model.add_layer(dense(392, 32, "relu"));
  model.add_layer(dense(32, 10, "softmax"));

  model.compile(LOSS_FUNCTION, OPTIMIZER, LEARNING_RATE);
  cout << "Modelo CNN compilado exitosamente." << endl;

  // Cargar datos (asegurarse que Reader::load_bin carga MNIST correctamente)
  vector<vector<float>> raw_X_train, raw_Y_train;
  Reader::load_bin("./database/mnist_train.bin", raw_X_train, raw_Y_train, 60000);

  // Validación y test
  vector<vector<float>> raw_X_test, raw_Y_test;
  Reader::load_bin("./database/mnist_test.bin", raw_X_test, raw_Y_test, 10000);
  cout << raw_X_train[0].size() << endl; // 784
  cout << raw_Y_train[0].size() << endl; // 10 one hot

  // Convertir a tensores
  vector<Tensor> X_train = to_tensor_batch_4D(raw_X_train);
  vector<Tensor> Y_train = to_tensor_batch_1D(raw_Y_train);
  vector<Tensor> X_test = to_tensor_batch_4D(raw_X_test);
  vector<Tensor> Y_test = to_tensor_batch_1D(raw_Y_test);

  // Entrenamiento
  auto start = start_timer();
  model.fit(X_train, Y_train, X_test, Y_test, EPOCHS, BATCH_SIZE, 1, true);
  double duration = stop_timer(start);
  print_duration(duration, "Tiempo de entrenamiento");

  // Save Model
  cout << "\nGuardando el modelo entrenado..." << endl;
  model.save_model("cnn_mnist.bin");

  test_model(model, X_test, Y_test);

  return 0;
}

void test_model(NeuralNetwork &model, vector<Tensor> &X_test, vector<Tensor> &Y_test) {
  int correct = 0;
  int total = X_test.size();

  for (int i = 0; i < total; i++) {
    Tensor pred = model.predict(X_test[i]);
    int pred_label = argmax(pred);
    int true_label = argmax(Y_test[i]);

    if (pred_label == true_label) {
      correct++;
    }
  }

  float accuracy = 100.0f * correct / total;
  cout << "Precisión en test: " << fixed << setprecision(2) << accuracy << "%" << endl;
}
