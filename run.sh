#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Debes proporcionar el nombre del ejecutable como parámetro."
    echo "Uso: ./run.sh <nombre_del_ejecutable>"
    exit 1
fi

EXECUTABLE=$1
INCLUDE_DIR="include"
TEST_DIR="test"

cd "$TEST_DIR" || { echo "Error: No se pudo cambiar al directorio '$TEST_DIR'"; exit 1; }

echo "--- Compilando $EXECUTABLE ---"
# el archivo fuente tiene el mismo nombre que el ejecutable con extension .cpp
g++ -I"../$INCLUDE_DIR" -o "$EXECUTABLE" "$EXECUTABLE.cpp" -std=c++11 # Puedes ajustar -std=c++11 si necesitas un estándar diferente
if [ $? -ne 0 ]; then
    echo "Error: La compilación de $EXECUTABLE falló."
    exit 1
fi

echo "--- Ejecutando $EXECUTABLE ---"
./"$EXECUTABLE"
if [ $? -ne 0 ]; then
    echo "Error: La ejecución de $EXECUTABLE falló."
    exit 1
fi

echo "--- $EXECUTABLE completado exitosamente ---"