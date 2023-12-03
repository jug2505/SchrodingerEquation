#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace std;

#define SQRT_M_PI 1.77245385091
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// Настройка SPH
#define N 100
constexpr double DT = 0.02;  // Шаг по времени
constexpr int NT = 100;  // Кол-во шагов по времени
constexpr int NT_SETUP = 400;  // Кол-во шагов на настройку
constexpr int N_OUT = 25;  // Вывод каждые N_OUT шагов
// i d_t psi + nabla^2/2 psi -x^2 psi/2 = 0
// Потенциал: 1/2 x^2
double b = 4;  // Демпфирование скорости для настройки начального состояния
#define M (1.0 / N) // Масса частицы SPH ( M * n = 1 normalizes |wavefunction|^2 to 1)
#define h (40.0 / N)  // Расстояние сглаживания
constexpr double xStart = -3.0;
constexpr double xEnd = 3.0;
constexpr double xStep = (xEnd - xStart) / (N - 1);

// Настройка CUDA
#define BLOCK_SIZE 32
#define GRID_SIZE ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)

// Коэффициенты задачи
#define m 7
#define gamma0 4.32e-12
#define chi 20.0
#define E0 0.0
#define omega 5.0e14
#define omega0 1.0e14
#define Kb 1.38e-16
#define T 77.0
#define a_eq 0.3
#define b_eq 2.0
#define F 0.0
#define F0 1.0

// Данные на GPU
double *x_dev, *xx_dev, *rho_dev, *drho_dev, *ddrho_dev, *P_dev, *u_dev, *a_dev;
// Данные на CPU
double *x, *u, *rho, *drho, *ddrho, *P, *a, *xx, *probe_rho, *u_mhalf, *u_phalf;

void init() {
    x = new double[N];
    u = new double[N];
    rho = new double[N];
    drho = new double[N];
    ddrho = new double[N];
    P = new double[N];
    a = new double[N];
    xx = new double[N];
    probe_rho = new double[N];
    u_mhalf = new double[N];
    u_phalf = new double[N];

    cudaMalloc(&x_dev, N * sizeof(double));
    cudaMalloc(&xx_dev, N * sizeof(double));
    cudaMalloc(&rho_dev, N * sizeof(double));
    cudaMalloc(&drho_dev, N * sizeof(double));
    cudaMalloc(&ddrho_dev, N * sizeof(double));
    cudaMalloc(&P_dev, N * sizeof(double));
    cudaMalloc(&u_dev, N * sizeof(double));
    cudaMalloc(&a_dev, N * sizeof(double));
    checkCudaErrors(cudaGetLastError());
}

void clear() {
    delete[] x;
    delete[] u;
    delete[] rho;
    delete[] drho;
    delete[] ddrho;
    delete[] P;
    delete[] a;
    delete[] xx;
    delete[] probe_rho;
    delete[] u_mhalf;
    delete[] u_phalf;

    cudaFree(x_dev);
    cudaFree(xx_dev);
    cudaFree(rho_dev);
    cudaFree(drho_dev);
    cudaFree(ddrho_dev);
    cudaFree(P_dev);
    cudaFree(u_dev);
    cudaFree(a_dev);
    checkCudaErrors(cudaGetLastError());
}

/* Гауссово сглаживающее ядро SPH (1D) */
__device__ double kernelDeriv0(double r) {
    return 1.0 / h / SQRT_M_PI * exp(- r * r / (h * h));
}
__device__ double kernelDeriv1(double r) {
    return pow(h, -3) / SQRT_M_PI * exp(- r * r / (h * h)) * (-2.0 * r);
}
__device__ double kernelDeriv2(double r) {
    return pow(h, -5) / SQRT_M_PI * exp(-r * r / (h * h)) * (4.0 * r * r - 2.0 * h * h);
}
__device__ double kernelDeriv3(double r) {
    return pow(h, -7) / sqrt(M_PI) * exp(-pow(r, 2) / pow(h, 2)) * (-8.0 * pow(r, 3) + 12.0 * pow(h, 2) * r);
}

__global__ void densityKernel(double* x, double* rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        sum += M * kernelDeriv0(uij);
    }
    rho[i] = sum;
}

/* Вычисление плотности в каждом из мест расположения частиц с помощью сглаживающего ядра */
__host__ void density(double* x, double* rho) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);

    densityKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, rho_dev);

    cudaMemcpy(rho, rho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void pressureKernelDRho(double* x, double* drho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        sum += M * kernelDeriv1(uij);
    }
    drho[i] = sum;
}

__global__ void pressureKernelDDRho(double* x, double* ddrho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        sum += M * kernelDeriv2(uij);
    }
    ddrho[i] = sum;
}

__global__ void pressureKernel(double* x, double* rho, double* drho, double* ddrho, double* P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        sum += 0.25 * (drho[j] * drho[j] / rho[j] - ddrho[j]) * M / rho[j] * kernelDeriv0(uij) / (chi * chi);
    }
    P[i] = sum;
}

/* Вычисление давления на каждой из частиц с помощью сглаживающего ядра
 * P = -(1/(4*chi^2))*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
 */
__host__ void pressure(double* x, double* rho, double* P) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_dev, rho, N * sizeof(double), cudaMemcpyHostToDevice);

    pressureKernelDRho<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, drho_dev);
    pressureKernelDDRho<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, ddrho_dev);

    cudaMemcpy(drho, drho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ddrho, ddrho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);

    pressureKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, rho_dev, drho_dev, ddrho_dev, P_dev);
    cudaMemcpy(P, P_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void accelerationKernel(double* x, double* u, double* rho, double* P, double b, double* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;


    // Нелинейность
    double P_NL = 0.0;
    if (b == 0) { // Включение нелинейности только после инициализации
        double * G_s_sum_array = new double[9] { 0.91, 0.327466, 0.1847448, 0.1182372, 0.0803538, 0.0565458, 0.0407006, 0.0297586, 0.0220082};
        double sum_nl = 0.0;
        for (int alpha = 1; alpha <= 9; alpha++) {
            double G_s_sum = G_s_sum_array[alpha - 1];
//        for (int s = 1; s <= m; s++) {
//            G_s_sum += G(alpha, s);
//        }
            double l_sum = 0.0;
            for (int l = 1; l <= 5; l++) {
                double f_l = 0.001; // TODO: Что это?
                l_sum += f_l * l / (l + 1.0) * pow(alpha, 2 * l) * pow(rho[i], l + 1);
            }
            sum_nl += alpha * G_s_sum * l_sum;
        }
        P_NL = 1.0 / (2.0 * chi * chi) * sum_nl;
        delete [] G_s_sum_array;
    }

    // Дэмпирование и гармонический потенциал (0.5 x^2)
    a[i] = - u[i] * b - x[i];

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        sum += -M * ((P[i] + P_NL) / (rho[i] * rho[i]) + P[j] / (rho[j] * rho[j])) * kernelDeriv1(uij);
    }

    a[i] = a[i] + sum;
    printf("%lf\n", a[i]);
}

/* Расчёт ускорения каждой частицы под действием квантового давления, гармонического потенциала, демпфирования скорости */
__host__ void acceleration(double* x, double* u, double* rho, double* P, double b, double* a) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_dev, rho, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u_dev, u, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(P_dev, P, N * sizeof(double), cudaMemcpyHostToDevice);

    accelerationKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, u_dev, rho_dev, P_dev, b, a_dev);

    cudaMemcpy(a, a_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void probeDensityKernel(double* x, double* xx, double* rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = xx[i] - x[j];
        sum += M * kernelDeriv0(uij);
    }
    rho[i] = sum;
}

/* Вычисление плотности в произвольных точках */
__host__ void probeDensity(double* x, double* xx, double* prob_rho) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(xx_dev, xx, N * sizeof(double), cudaMemcpyHostToDevice);

    probeDensityKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, xx_dev, rho_dev);

    cudaMemcpy(prob_rho, rho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

void getCudaInfo() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("Number of devices: %d\n", nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
               prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
    }
    cudaSetDevice(0);
}


void compute() {
    getCudaInfo();
    init();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Инициализация положений и скоростей частиц
    for (int i = 0; i < N; i++) {
        x[i] = xStart + i * xStep;
        xx[i] = x[i]; // Для графика
    }

    // Инициализация плотности, давления и ускорения
    density(x, rho);
    pressure(x, rho, P);
    acceleration(x, u, rho, P, b, a);

    // v в t=-0.5*DT для leap frog интегратора
    for (int i = 0; i < N; i++) {
        u_mhalf[i] = u[i] - 0.5 * DT * a[i];
    }

    ofstream outfile("solution_cuda.txt");
    outfile << "X Z" << endl;

    // Главный цикл по времени
    double t = 0.0;
    for (int i = -NT_SETUP; i < NT; i++) {
        // Leap frog
        for (int j = 0; j < N; j++) {
            u_phalf[j] = u_mhalf[j] + a[j] * DT;
            x[j] = x[j] + u_phalf[j] * DT;
            u[j] = 0.5 * (u_mhalf[j] + u_phalf[j]);
            u_mhalf[j] = u_phalf[j];
        }

        if (i >= 0) {
            t = t + DT;
        }
        cout << "SPH: t = " << t << endl;

        if (i == -1) {
            for (int j = 0; j < N; j++) {
                u_mhalf[j] = 1.0;
            }
            b = 0.0;
        }

        // Обновление плотностей, давлений, ускорений
        density(x, rho);
        pressure(x, rho, P);
        acceleration(x, u, rho, P, b, a);

        // Вывод в файлы
        if (i >= 0 && i % N_OUT == 0) {
            probeDensity(x,xx, probe_rho);
            for (int j = 0; j < N; j++) {
                outfile << xx[j] << " " << probe_rho[j] << endl;
            }
        }
    }
    outfile.close();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Время работы: %3.1f s\n", elapsedTime / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    clear();
}


int main() {
    compute();
    return 0;
}
