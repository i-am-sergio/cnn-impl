#pragma once
#include "Layer.hpp"
#include "Dense.hpp"
#include "Dropout.hpp"
#include "Optimizer.hpp"
#include "Utils.hpp"

#include <memory>
#include <fstream>
#include <stdexcept>
#include <vector>

class NeuralNetwork
{
private:
    vector<unique_ptr<Layer>> layers; // Vector de capas de la red
    unique_ptr<Optimizer> optimizer;  // Puntero al optimizador
    string error_function;            // funcion para calculo del error
public:
    NeuralNetwork(string error_function = "cross-entropy")
    {
        this->error_function = error_function;
    }

    // Funcion para configurar el optimizador y la tasa de aprendizaje
    void compile(const string &loss_function = "cross-entropy", const string &optimizer_name = "sgd",
                 float learning_rate = 0.001, float beta1 = 0.9f, float beta2 = 0.999f)
    {
        // cout << "- Optimizador Usado: " << optimizer_name << endl;
        // cout << "- Funcion de Perdida: " << loss_function << endl;

        this->error_function = loss_function;

        if (optimizer_name == "sgd")
        {
            optimizer = make_unique<SGD_Optimizer>(learning_rate);
        }
        else if (optimizer_name == "rmsprop")
        {
            optimizer = make_unique<RMSProp_Optimizer>(learning_rate, beta1);
        }
        else if (optimizer_name == "adam")
        {
            optimizer = make_unique<Adam_Optimizer>(learning_rate, beta1, beta2);
        }
        else
        {
            throw runtime_error("Optimizador no soportado: " + optimizer_name);
        }

        if (loss_function != "mse" && loss_function != "cross-entropy")
        {
            throw runtime_error("Funcion de perdida no soportada: " + loss_function);
        }
    }

    void add_layer(unique_ptr<Layer> layer)
    {                                       // Agrega capa a la red
        layers.push_back(std::move(layer)); // Inserta usando move semantics
    }

    // Calcula error cuadratico medio
    float mse(const Tensor &y_pred, const Tensor &y_true) const
    {
        float sum = 0.0f;
        for (size_t i = 0; i < y_true.shape[0]; ++i)
        {
            float diff = y_pred({i}) - y_true({i}); // Diferencia entre prediccion y real
            sum += diff * diff;                     // Suma cuadrado de la diferencia
        }
        return sum / y_true.shape[0]; // Retorna promedio
    }

    // Derivada de MSE
    Tensor mse_derivative(const Tensor &y_pred, const Tensor &y_true) const
    {
        Tensor grad({y_true.shape[0]}); // Gradiente del mismo tamaño
        for (size_t i = 0; i < y_true.shape[0]; ++i)
            grad({i}) = 2.0f * (y_pred({i}) - y_true({i})) / y_true.shape[0];
        return grad; // Retorna Derivada del error (perdida)
    }

    float cross_entropy(const Tensor &pred, const Tensor &target) const
    {
        float loss = 0.0f;
        const auto &p = pred.data;
        const auto &t = target.data;
        for (size_t i = 0; i < p.size(); ++i)
            loss -= t[i] * std::log(std::max(p[i], 1e-8f)); // evitar log(0)
        return loss;                                        // perdida escalar por muestra
    }

    Tensor cross_entropy_derivative(const Tensor &pred, const Tensor &target) const
    {
        const auto &p = pred.data;
        const auto &t = target.data;
        Tensor grad(pred.shape);
        auto &g = grad.data;
        for (size_t i = 0; i < p.size(); ++i)
        {
            g[i] = p[i] - t[i]; // simplificacion de softmax + cross-entropy
        }
        return grad;
    }

    float compute_total_loss(const Tensor &y_pred, const Tensor &y_true) const
    {
        float loss = 0.0f;

        // Perdida original
        if (error_function == "cross-entropy")
        {
            loss = cross_entropy(y_pred, y_true);
        }
        else
        {
            loss = mse(y_pred, y_true);
        }

        // Termino L2 (Weight Decay) de todas las capas
        float l2_term = 0.0f;
        for (const auto &layer : layers)
        {
            if (auto dense_layer = dynamic_cast<Dense *>(layer.get()))
            {
                for (float w : dense_layer->weights.data)
                {
                    l2_term += w * w;
                }
            }
        }

        return loss + l2_term;
    }

    float accuracy(const Tensor &y_pred, const Tensor &y_true) const
    {
        int pred_class = argmax(y_pred);
        int true_class = argmax(y_true);
        return (pred_class == true_class) ? 1.0f : 0.0f;
    }

    Tensor forward(const Tensor &input) const
    {
        Tensor out = input;              // Salida inicial es la entrada
        for (const auto &layer : layers) // Itera sobre cada capa
            out = layer->forward(out);   // Pasa la salida a la siguiente capa
        return out;                      // Retorna la salida final
    }

    // Entrenamiento con multiples ejemplos por varias epocas
    void fit(const vector<Tensor> &X, const vector<Tensor> &Y,
             const vector<Tensor> &X_valid, const vector<Tensor> &Y_valid,
             int epochs, int batch_size = 1, int verbose_every = 1000, bool training_logs = false)
    {

        if (!optimizer)
            throw std::runtime_error("Modelo no compilado. Llamar a 'compile()' primero.");

        std::ofstream log_file;
        if (training_logs)
        {
            log_file.open("log_" + to_string(epochs) + "ep.txt");
            log_file << "Epoch,Train_Loss,Train_Accuracy,Valid_Loss,Valid_Accuracy\n";
        }

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            // Entrenamiento
            auto start = start_timer();
            float total_train_loss = 0.0f;
            float total_train_accuracy = 0.0f;
            int num_batches = (X.size() + batch_size - 1) / batch_size;

            // Modo entrenamiento para Dropout
            for (const auto &layer : layers)
            {
                if (auto dropout_layer = dynamic_cast<Dropout *>(layer.get()))
                {
                    dropout_layer->set_training_mode(true);
                }
            }

            for (int batch_idx = 0; batch_idx < num_batches; batch_idx++)
            {
                // Calcular indices del batch actual
                int start_idx = batch_idx * batch_size;
                int end_idx = min(start_idx + batch_size, (int)X.size());

                // Procesar batch (forward + backward)
                float batch_loss = 0.0f; // Perdida original
                float batch_accuracy = 0.0f;
                float batch_l2 = 0.0f; // Termino L2 acumulado

                // 1. Forward pass y calculo de perdida base
                for (int i = start_idx; i < end_idx; i++)
                {
                    Tensor pred = forward(X[i]);

                    if (error_function == "cross-entropy")
                    {
                        batch_loss += cross_entropy(pred, Y[i]);
                    }
                    else
                    {
                        batch_loss += mse(pred, Y[i]);
                    }
                    batch_accuracy += accuracy(pred, Y[i]);
                }

                // 2. Calcular termino L2 de todas las capas Dense
                for (const auto &layer : layers)
                {
                    if (auto dense_layer = dynamic_cast<Dense *>(layer.get()))
                    {
                        batch_l2 += dense_layer->compute_l2_penalty();
                    }
                }

                for (auto &layer : layers)
                {
                    layer->zero_grad(); // <<<<<< INICIALIZA acumuladores en cero
                }

                // 3. Backward pass (igual que antes)
                for (int i = start_idx; i < end_idx; i++)
                {
                    Tensor grad = (error_function == "cross-entropy")
                                      ? cross_entropy_derivative(forward(X[i]), Y[i])
                                      : mse_derivative(forward(X[i]), Y[i]);

                    for (int j = layers.size() - 1; j >= 0; j--)
                    {
                        grad = layers[j]->backward(grad);
                    }
                }

                // 4. Normalizar gradientes (dividir entre batch_size) si necesario
                int current_batch_size = end_idx - start_idx;
                for (auto &layer : layers)
                {
                    if (auto dense_layer = dynamic_cast<Dense *>(layer.get()))
                    {
                        dense_layer->scale_gradients(1.0f / current_batch_size); // <--
                    }
                }

                // 5. Actualizar parametros
                for (auto &layer : layers)
                {
                    layer->update_parameters(*optimizer);
                }

                // 5. Acumular metricas (perdida promedio del batch + L2)
                // total_train_loss += (batch_loss / batch_size) + batch_l2;
                total_train_loss += (batch_loss + batch_l2) / current_batch_size;
                total_train_accuracy += batch_accuracy;
            }
            double duration = stop_timer(start);

            // Validacion
            float total_valid_loss = 0.0f;
            float total_valid_accuracy = 0.0f;

            for (const auto &layer : layers)
            {
                if (auto dropout_layer = dynamic_cast<Dropout *>(layer.get()))
                {
                    dropout_layer->set_training_mode(false);
                }
            }

            for (size_t i = 0; i < X_valid.size(); i++)
            {
                Tensor pred = predict(X_valid[i]);
                if (error_function == "cross-entropy")
                    total_valid_loss += cross_entropy(pred, Y_valid[i]);
                else
                    total_valid_loss += mse(pred, Y_valid[i]);
                total_valid_accuracy += accuracy(pred, Y_valid[i]);
            }

            // Calcular promedios
            float avg_train_loss = total_train_loss / X.size() * batch_size;
            float avg_train_acc = total_train_accuracy / X.size();
            float avg_valid_loss = total_valid_loss / X_valid.size();
            float avg_valid_acc = total_valid_accuracy / X_valid.size();

            // Logging
            if (verbose_every > 0 && (epoch % verbose_every == 0 || epoch == epochs))
            {
                cout << BOLD << CYAN << "─ Epoch " << epoch << RESET << " (Batch: " << batch_size << ")\n";
                cout << "  " << GREEN << "Train Loss: " << fixed << setprecision(4) << avg_train_loss;
                cout << "  " << GREEN << "Train Acc:  " << fixed << setprecision(4) << avg_train_acc << RESET;
                cout << "  " << MAGENTA << "Valid Loss: " << fixed << setprecision(4) << avg_valid_loss;
                cout << "  " << MAGENTA << "Valid Acc:  " << fixed << setprecision(4) << avg_valid_acc << RESET;
                cout << "  " << YELLOW << "Time: " << fixed << setprecision(2) << duration << "s" << RESET << "\n";
                cout << BOLD << CYAN << "──────────────────────\n"
                     << RESET;
            }
            if (training_logs)
            {
                log_file << epoch << ","
                         << avg_train_loss << "," << avg_train_acc << ","
                         << avg_valid_loss << "," << avg_valid_acc << "\n";
            }
        }

        if (training_logs)
            log_file.close();
    }

    // Realizar una prediccion con la red neuronal
    Tensor predict(const Tensor &input) const
    {
        // Todas las capas Dropout deben estar en modo de inferencia
        for (const auto &layer : layers)
        {
            if (auto dropout_layer = dynamic_cast<Dropout *>(layer.get()))
            {
                dropout_layer->set_training_mode(false);
            }
        }
        return forward(input);
    }
};