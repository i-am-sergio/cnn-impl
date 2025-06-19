#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>

using namespace std;

class Reader {
public:
    static void load_csv(const string& filename, vector<vector<float>>& X, vector<vector<float>>& Y, size_t rows = -1) {
        ifstream file(filename);
        string line;
        size_t row_count = 0;

        while (getline(file, line)) {
            if (rows > 0 && row_count >= rows)
                break;

            stringstream ss(line);
            string token;
            vector<float> row;

            while (getline(ss, token, ',')) {
                row.push_back(stof(token));
            }

            if (row.size() != 784 + 10) {
                cerr << "Fila inválida con " << row.size() << " columnas: " << line << endl;
                continue;
            }

            vector<float> x(row.begin(), row.begin() + 784);   // primeras 784 columnas
            vector<float> y(row.begin() + 784, row.end());     // últimas 10 columnas (one-hot)

            X.push_back(x);
            Y.push_back(y);

            row_count++;
        }

        file.close();
    }

    static void load_bin(const string& filename, vector<vector<float>>& X, vector<vector<float>>& Y, size_t max_rows = -1) {
        ifstream file(filename, ios::binary);
        if (!file.is_open()) {
            throw runtime_error("No se pudo abrir el archivo binario: " + filename);
        }

        // Leer cabecera
        int32_t header[3];
        file.read(reinterpret_cast<char*>(header), sizeof(header));
        
        int num_images = header[0];
        int rows = header[1];
        int cols = header[2];
        int image_size = rows * cols;

        if (max_rows > 0 && max_rows < static_cast<size_t>(num_images)) {
            num_images = static_cast<int>(max_rows);
        }

        X.reserve(num_images);
        Y.reserve(num_images);

        for (int i = 0; i < num_images; ++i) {
            // Leer etiqueta
            unsigned char label;
            file.read(reinterpret_cast<char*>(&label), 1);

            // Leer imagen
            vector<unsigned char> pixels(image_size);
            file.read(reinterpret_cast<char*>(pixels.data()), image_size);

            // Convertir a float y normalizar [0, 1]
            vector<float> x;
            x.reserve(image_size);
            for (auto pixel : pixels) {
                x.push_back(static_cast<float>(pixel) / 255.0f);
            }

            // Crear one-hot encoding
            vector<float> y(10, 0.0f);
            y[label] = 1.0f;

            X.push_back(move(x));
            Y.push_back(move(y));
        }

        file.close();
    }
};