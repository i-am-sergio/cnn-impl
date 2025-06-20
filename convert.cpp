// #include <fstream>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <cassert>
// #include <iomanip>

// using namespace std;

// void convert_mnist_to_csv(const string& image_file, const string& label_file, const string& output_csv, int limit = -1) {
//     ifstream images(image_file, ios::binary);
//     ifstream labels(label_file, ios::binary);
//     ofstream output(output_csv);

//     assert(images.is_open() && "No se pudo abrir el archivo de imagenes");
//     assert(labels.is_open() && "No se pudo abrir el archivo de etiquetas");

//     // Leer cabecera de imagenes
//     int32_t magic_images = 0, num_images = 0, rows = 0, cols = 0;
//     images.read(reinterpret_cast<char*>(&magic_images), 4);
//     images.read(reinterpret_cast<char*>(&num_images), 4);
//     images.read(reinterpret_cast<char*>(&rows), 4);
//     images.read(reinterpret_cast<char*>(&cols), 4);

//     // Leer cabecera de etiquetas
//     int32_t magic_labels = 0, num_labels = 0;
//     labels.read(reinterpret_cast<char*>(&magic_labels), 4);
//     labels.read(reinterpret_cast<char*>(&num_labels), 4);

//     // Convertir endian
//     magic_images = __builtin_bswap32(magic_images);
//     num_images = __builtin_bswap32(num_images);
//     rows = __builtin_bswap32(rows);
//     cols = __builtin_bswap32(cols);
//     magic_labels = __builtin_bswap32(magic_labels);
//     num_labels = __builtin_bswap32(num_labels);

//     assert(magic_images == 2051);
//     assert(magic_labels == 2049);
//     assert(num_images == num_labels);

//     int image_size = rows * cols;
//     unsigned char pixel;
//     unsigned char label;

//     if (limit < 0 || limit > num_images) limit = num_images;

//     for (int i = 0; i < limit; ++i) {
//         // Leer imagen
//         vector<float> pixels(image_size);
//         for (int j = 0; j < image_size; ++j) {
//             images.read(reinterpret_cast<char*>(&pixel), 1);
//             pixels[j] = pixel / 255.0f;  // Normalizar a [0,1]
//         }

//         // Leer etiqueta
//         labels.read(reinterpret_cast<char*>(&label), 1);

//         // Escribir pixeles optimizando los ceros
//         for (int j = 0; j < image_size; ++j) {
//             if (pixels[j] == 0.0f) {
//                 output << "0";
//             } else {
//                 output << fixed << setprecision(6) << pixels[j];
//             }
//             if (j < image_size - 1) output << ",";
//         }

//         // Escribir salida one-hot
//         output << ",";
//         for (int j = 0; j < 10; ++j) {
//             output << (j == label ? "1" : "0");
//             if (j < 9) output << ",";
//         }

//         output << "\n";
//     }

//     cout << "Archivo CSV generado con " << limit << " ejemplos: " << output_csv << endl;
// }

// int main() {
//     convert_mnist_to_csv(
//         "train-images.idx3-ubyte",
//         "train-labels.idx1-ubyte",
//         "mnist_train_flat_60000.csv",
//          60000
//     );

//     convert_mnist_to_csv(
//         "t10k-images.idx3-ubyte",
//         "t10k-labels.idx1-ubyte",
//         "mnist_test_flat_10000.csv",
//         10000
//     );

//     return 0;
// }

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>

using namespace std;

struct MnistData
{
    unsigned char label;
    vector<unsigned char> pixels;
};

void convert_mnist_to_bin(const string &image_file, const string &label_file,
                          const string &output_bin, int limit = -1)
{
    ifstream images(image_file, ios::binary);
    ifstream labels(label_file, ios::binary);
    ofstream output(output_bin, ios::binary);

    assert(images.is_open() && "No se pudo abrir el archivo de imagenes");
    assert(labels.is_open() && "No se pudo abrir el archivo de etiquetas");
    assert(output.is_open() && "No se pudo crear el archivo de salida");

    // Leer cabeceras
    int32_t magic_images, num_images, rows, cols;
    int32_t magic_labels, num_labels;

    images.read(reinterpret_cast<char *>(&magic_images), 4);
    images.read(reinterpret_cast<char *>(&num_images), 4);
    images.read(reinterpret_cast<char *>(&rows), 4);
    images.read(reinterpret_cast<char *>(&cols), 4);

    labels.read(reinterpret_cast<char *>(&magic_labels), 4);
    labels.read(reinterpret_cast<char *>(&num_labels), 4);

    // Convertir endianness
    magic_images = __builtin_bswap32(magic_images);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);
    magic_labels = __builtin_bswap32(magic_labels);
    num_labels = __builtin_bswap32(num_labels);

    assert(magic_images == 2051 && "Formato incorrecto de archivo de imágenes");
    assert(magic_labels == 2049 && "Formato incorrecto de archivo de etiquetas");
    assert(num_images == num_labels && "Número de imágenes y etiquetas no coincide");

    if (limit < 0 || limit > num_images)
        limit = num_images;

    const int image_size = rows * cols;
    vector<MnistData> dataset(limit);

    // Leer todos los datos
    for (int i = 0; i < limit; ++i)
    {
        dataset[i].pixels.resize(image_size);
        images.read(reinterpret_cast<char *>(dataset[i].pixels.data()), image_size);
        labels.read(reinterpret_cast<char *>(&dataset[i].label), 1);
    }

    // Escribir archivo binario compacto
    // Formato: [header][data]
    // header: (int32) num_images, (int32) rows, (int32) cols
    // data: para cada imagen: (uchar) label, (uchar[rows*cols]) pixels

    // Escribir cabecera
    int32_t header[3] = {limit, rows, cols};
    output.write(reinterpret_cast<char *>(header), sizeof(header));

    // Escribir datos
    for (const auto &item : dataset)
    {
        output.write(reinterpret_cast<const char *>(&item.label), 1);
        output.write(reinterpret_cast<const char *>(item.pixels.data()), image_size);
    }

    cout << "Archivo binario generado: " << output_bin << endl;
    cout << "Número de ejemplos: " << limit << endl;
    cout << "Dimensiones de imagen: " << rows << "x" << cols << endl;
    cout << "Tamaño del archivo: " << (sizeof(header) + limit * (1 + image_size))
         << " bytes" << endl;
}

int main()
{
    // Convertir conjunto de entrenamiento
    convert_mnist_to_bin(
        "./archives/train-images.idx3-ubyte",
        "./archives/train-labels.idx1-ubyte",
        "./database/mnist_train.bin",
        60000);

    // Convertir conjunto de prueba
    convert_mnist_to_bin(
        "./archives/t10k-images.idx3-ubyte",
        "./archives/t10k-labels.idx1-ubyte",
        "./database/mnist_test.bin",
        10000);

    return 0;
}