#pragma once

#include <chrono>
#include <memory>
#include <vector>
#include <iomanip>

using namespace std;

#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"
#define MAGENTA "\033[35m"

// Cronometro
using Time = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Time>;

// Inicia el cronometro
inline TimePoint start_timer() {
    return Time::now();
}

// Detiene el cronometro y devuelve la duracion en segundos
inline double stop_timer(const TimePoint& start_time) {
    TimePoint end_time = Time::now();
    std::chrono::duration<double> duration = end_time - start_time;
    return duration.count();
}

// Imprime duracion en segundos con formato
inline void print_duration(double duration, const string& label) {
    cout << label << ": " << fixed << setprecision(2) << duration << " s" << endl;
}
