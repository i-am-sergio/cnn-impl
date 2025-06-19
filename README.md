# Implementación de una CNN en C++ desde cero

Este proyecto implementa una red neuronal convolucional (CNN) en C++, con clases para manejar tensores, capas y operaciones matemáticas necesarias.

## Estructura de Clases

### `Tensor` (Tensor.hpp)
- **Responsabilidad**: Almacenar y manipular datos multidimensionales.
- **Características**:
  - Almacenamiento en vector lineal con formas y strides.
  - Cálculo automático de offsets para acceso multidimensional.
  - Operaciones básicas de indexación y manipulación de formas.

### `Layer` (Layer.hpp)
- **Clase base abstracta** para todas las capas de la red.
- **Métodos virtuales puros**:
  - `forward()`: Propagación hacia adelante
  - `backward()`: Propagación hacia atrás (gradientes)
  - `update_parameters()`: Actualización de pesos
  - `print()`: Visualización de la capa
  - `zero_grad()`: Reinicio de gradientes

### Capas Implementadas

#### `Dense` (Capa Fully Connected)
- **Características**:
  - Pesos (`weights`) y sesgos (`bias`)
  - Soporte para diferentes funciones de activación
  - Cálculo y almacenamiento de gradientes
  - Weight decay (regularización L2)

#### `Conv2D` (Capa Convolucional 2D)
- **Características**:
  - Filtros convolucionales
  - Padding y stride configurable
  - Cálculo eficiente de gradientes

### Clases de Soporte

#### `Math` (Math.hpp)
- **Operaciones matemáticas**:
  - Multiplicación de tensores (`dot_product`)
  - Operaciones convolucionales
  - Funciones de activación y sus derivadas

#### `Utils` (Utils.hpp)
- Funciones auxiliares:
  - Normalización de datos
  - Codificación one-hot
  - Split train-test

#### `Reader` (Reader.hpp)
- **Carga de datos**:
  - Lectura de archivos CSV
  - Parseo a tensores
  - Separación automática features/labels

### `CNN` (CNN.hpp)
- **Clase principal** que ensambla la red:
  - Gestión del ciclo de entrenamiento
  - Secuencia de capas
  - Optimizador configurable
  - Cálculo de pérdida

## Flujo de Trabajo

1. **Inicialización**:
   ```cpp
   CNN model;
   model.add_layer(make_unique<Conv2D>(...));
   model.add_layer(make_unique<Dense>(...));
   model.set_optimizer(make_unique<SGD>(0.01));
   ```

2. **Entrenamiento**:
   ```cpp
   auto [X_train, Y_train] = Reader::load_csv("data.csv");
   model.train(X_train, Y_train, epochs=10);
   ```

3. **Predicción**:
   ```cpp
   Tensor output = model.predict(input_tensor);
   ```

## Compilación
Requiere C++17 y OpenMP para paralelización:
```bash
g++ -std=c++17 -fopenmp main.cpp -o main
```

## Dependencias
- Solo biblioteca estándar de C++17
- OpenMP para operaciones paralelizadas
