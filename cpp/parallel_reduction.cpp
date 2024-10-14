#include <iostream>
#include <chrono>  // For timing in C++

int main() {
    const int N = 1000000000;  // 1 Billion
    int *arr = new int[N];  // Dynamically allocated array

    // Initialize array with random values
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 1000;  // Random values between 0 and 999
    }

    // Measure execution time for C++ reduction
    auto cpp_start = std::chrono::high_resolution_clock::now();

    // Compute sum
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }

    auto cpp_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpp_duration = cpp_end - cpp_start;

    std::cout << "C++ Sum: " << sum << std::endl;
    std::cout << "C++ Execution Time: " << cpp_duration.count() << " seconds" << std::endl;

    delete[] arr;  // Free dynamically allocated memory

    return 0;
}
