#include "Conv2D.hpp"
#include "Dense.hpp"
#include "Flatten.hpp"
#include "Layer.hpp"
#include "NeuralNetwork.hpp"
#include "Pool2D.hpp"
#include "Reader.hpp"
#include "Tensor.hpp"
#include "Utils.hpp"

#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

void test_model(NeuralNetwork &model, vector<Tensor> &X_test, vector<Tensor> &Y_test) {
  cout << "\nIniciando evaluación del modelo en el conjunto de test..." << endl;
  int correct = 0;
  const int total = X_test.size();

  for (int i = 0; i < total; i++) {
    Tensor pred = model.predict(X_test[i]);

    int pred_label = argmax(pred);
    int true_label = argmax(Y_test[i]);

    if (pred_label == true_label) {
      correct++;
    }
  }

  // Calcula y muestra la precisión final
  float accuracy = 100.0f * static_cast<float>(correct) / static_cast<float>(total);
  cout << "Evaluación completada." << endl;
  cout << " -> Precisión final en el conjunto de test: " << fixed << setprecision(2) << accuracy << "% (" << correct << "/"
       << total << " correctas)" << endl;
}

int main() {
  cout << "--- Test de Modelo CNN Pre-entrenado ---" << endl;

  // Reconstruir la arquitectura del modelo
  cout << "1. Reconstruyendo la arquitectura del modelo..." << endl;
  NeuralNetwork model;

  model.add_layer(conv2d(1, 8, 5, 2, 2));
  model.add_layer(pool(2, 2, PoolingType::MAX));
  model.add_layer(flatten());
  model.add_layer(dense(392, 32, "relu"));
  model.add_layer(dense(32, 10, "softmax"));

  cout << "   Arquitectura creada." << endl;

  // Cargar los pesos y sesgos desde el archivo binario
  const string model_path = "models/cnn_mnist.bin";
  cout << "2. Cargando pesos desde '" << model_path << "'..." << endl;
  try {
    model.load_model(model_path);
    cout << "   Modelo cargado exitosamente." << endl;
  } catch (const runtime_error &e) {
    cerr << "Error crítico: " << e.what() << endl;
    cerr << "Asegúrate de haber ejecutado primero el programa de entrenamiento (cnn.cpp) para generar '" << model_path
         << "' en el directorio correcto." << endl;
    return 1;
  }

  // Cargar los datos de test de MNIST
  cout << "3. Cargando datos de test de MNIST..." << endl;
  vector<vector<float>> raw_X_test, raw_Y_test;
  Reader::load_bin("./database/mnist_test.bin", raw_X_test, raw_Y_test, 10000);

  vector<Tensor> X_test = to_tensor_batch_4D(raw_X_test);
  vector<Tensor> Y_test = to_tensor_batch_1D(raw_Y_test);
  cout << "   Datos de test cargados (" << X_test.size() << " muestras)." << endl;

  // Evaluar el modelo con los datos cargados
  test_model(model, X_test, Y_test);

  return 0;
}
