#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

using namespace std;

// Для BLAS
extern "C" {
    void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* lda, const double* b, const int* ldb,
        const double* beta, double* c, const int* ldc);
}

void matrixMultiply(const vector<double>& A, const vector<double>& B,
    vector<double>& C, int n) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}


void matrixMultiplyOptimized(const vector<double>& A, const vector<double>& B,
    vector<double>& C, int n) {
    const int block_size = 128; 
    const int unroll_factor = 4; 

#pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            double* __restrict c_ptr = &C[ii * n + jj];
            for (int kk = 0; kk < n; kk += block_size) {
                const double* __restrict a_ptr = &A[ii * n + kk];
                const double* __restrict b_ptr = &B[kk * n + jj];

                for (int i = 0; i < block_size; i += unroll_factor) {
                    for (int k = 0; k < block_size; ++k) {
                        double a0 = a_ptr[i * n + k];
                        double a1 = a_ptr[(i + 1) * n + k];
                        double a2 = a_ptr[(i + 2) * n + k];
                        double a3 = a_ptr[(i + 3) * n + k];

                        double* c0 = &c_ptr[i * n];
                        double* c1 = &c_ptr[(i + 1) * n];
                        double* c2 = &c_ptr[(i + 2) * n];
                        double* c3 = &c_ptr[(i + 3) * n];

                        for (int j = 0; j < block_size; ++j) {
                            double b_val = b_ptr[k * n + j];
                            c0[j] += a0 * b_val;
                            c1[j] += a1 * b_val;
                            c2[j] += a2 * b_val;
                            c3[j] += a3 * b_val;
                        }
                    }
                }
            }
        }
    }
}

void generate_random_matrix(vector<double>& matrix, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);

#pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = dis(gen);
    }
}

int main() {

    setlocale(LC_ALL, "rus");

    const int n = 4096;
    const double complexity = 2.0 * n * n * n;

    vector<double> A(n * n);
    vector<double> B(n * n);
    vector<double> C(n * n, 0.0);

    cout << "Создание случайных матриц..." << endl;
    generate_random_matrix(A, n);
    generate_random_matrix(B, n);

    cout << "1. Стандартный вариант перемножения..." << endl;
    auto start = chrono::high_resolution_clock::now();
    matrixMultiply(A, B, C, n);
    auto end = chrono::high_resolution_clock::now();
    double time_basic = chrono::duration<double>(end - start).count();
    double mflops_basic = (complexity / time_basic) * 1e-6;

    cout << "Время: " << time_basic << " с, Производительность: " << mflops_basic << " MFlops\n" << endl;
    fill(C.begin(), C.end(), 0.0);

    cout << "2. Перемножение с использованием BLAS..." << endl;
    char trans = 'N';
    double alpha = 1.0;
    double beta = 0.0;

    start = chrono::high_resolution_clock::now();
    dgemm_(&trans, &trans, &n, &n, &n, &alpha, A.data(), &n, B.data(), &n, &beta, C.data(), &n);
    end = chrono::high_resolution_clock::now();
    double time_blas = chrono::duration<double>(end - start).count();
    double mflops_blas = (complexity / time_blas) * 1e-6;

    cout << "Время: " << time_blas << " с, Производительность: " << mflops_blas << " MFlops\n" << endl;
    fill(C.begin(), C.end(), 0.0);

    cout << "3. Оптимизированный алгоритм перемножения..." << endl;
    start = chrono::high_resolution_clock::now();
    matrixMultiplyOptimized(A, B, C, n);
    end = chrono::high_resolution_clock::now();
    double time_opt = chrono::duration<double>(end - start).count();
    double mflops_opt = (complexity / time_opt) * 1e-6;

    cout << "Время: " << time_opt << " с, Производительность: " << mflops_opt << " MFlops\n" << endl;

    return 0;
}