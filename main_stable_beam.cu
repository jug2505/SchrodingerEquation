#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>

using namespace std;

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#define SQRT_M_PI 1.77245385091

enum class Type{ FLEX, SOLID };

// Константы CUDA
#define BLOCK_SIZE 32

// Константы SPH
#define N 100
#define SOLID_LAYER_LENGTH 3
constexpr double DT = 0.02;  // Шаг по времени
constexpr int NT = 1500;  // Кол-во шагов по времени
constexpr int NT_SETUP = 0;  // Кол-во шагов на настройку
constexpr int N_OUT = 15;  // Вывод каждые N_OUT шагов
constexpr int N_PROGRESS = 10;
constexpr int PROGRESS_STEP = NT / N_PROGRESS;

// i d_t psi + nabla^2/2 psi -x^2 psi/2 = 0
// Потенциал: 1/2 x^2
double b = 0;  // Демпфирование скорости для настройки начального состояния
#define M (1.0 / N) // Масса частицы SPH ( M * n = 1 normalizes |wavefunction|^2 to 1)
#define H_DEFAULT (0.4)  // Расстояние сглаживания
#define H_COEF 8
constexpr double xStart = -3.0;
constexpr double xEnd = 3.0;
constexpr double xStep = (xEnd - xStart) / (N - 1);

// Коэффициенты задачи
double gamma0 = 4.32e-12;
double Kb = 1.38e-16;
double T = 77.0;
// chi=20 a=0.0065
// chi=40 a=0.0323
// chi=4 a=0.323
#define chi 20
double a_eq = 0.0065;
double b_eq = 2.0;
int m = 7;
#define ALPHA_MAX 9
#define L_MAX 9
double R = (-0.25*gamma0); // R = Q = -D = 0 , (-0.25*gamma0), (-0.5*gamma0)
double Q = R;
double D = -Q;

// Кол-во разбиений для интеграла
const int num_splits = 1000;

// Кэш
map<pair<int, int>, double> G_alpha_s_cache;
map<pair<int, int>, double> delta_alpha_s_cache;
map<int, double> G_alpha_cache;

// На GPU
double* x_dev;
double* xx_dev;
double* rho_dev;
double* drho_dev;
double* ddrho_dev;
double* P_dev;
double* u_dev;
double* a_dev;
double* mass_dev;
double* h_array_dev;
Type* particles_type_dev;
double* G_s_sum_array_dev;

// На CPU
double* x;
double* u;
double* rho;
double* drho;
double* ddrho;
double* P;
double* a;
double* xx; // Для графика
double* probe_rho; // Для графика
double* u_mhalf;
double* u_phalf;
double* mass;
double* h_array;
Type* particles_type;
double* G_s_sum_array;



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
    mass = new double[N];
    h_array = new double[N];
    particles_type = new Type[N];
    G_s_sum_array = new double[ALPHA_MAX];


    cudaMalloc(&x_dev, N * sizeof(double));
    cudaMalloc(&xx_dev, N * sizeof(double));
    cudaMalloc(&rho_dev, N * sizeof(double));
    cudaMalloc(&drho_dev, N * sizeof(double));
    cudaMalloc(&ddrho_dev, N * sizeof(double));
    cudaMalloc(&P_dev, N * sizeof(double));
    cudaMalloc(&u_dev, N * sizeof(double));
    cudaMalloc(&a_dev, N * sizeof(double));
    cudaMalloc(&mass_dev, N * sizeof(double));
    cudaMalloc(&h_array_dev, N * sizeof(double));
    cudaMalloc(&particles_type_dev, N * sizeof(Type));
    cudaMalloc(&G_s_sum_array_dev, ALPHA_MAX * sizeof(double));
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
    delete[] mass;
    delete[] h_array;
    delete[] particles_type;
    delete[] G_s_sum_array;

    cudaFree(x_dev);
    cudaFree(xx_dev);
    cudaFree(rho_dev);
    cudaFree(drho_dev);
    cudaFree(ddrho_dev);
    cudaFree(P_dev);
    cudaFree(u_dev);
    cudaFree(a_dev);
    cudaFree(mass_dev);
    cudaFree(h_array_dev);
    cudaFree(particles_type_dev);
    cudaFree(G_s_sum_array_dev);
    checkCudaErrors(cudaGetLastError());
}


__host__ __device__ double factorial(const int n) {
    double f = 1;
    for (int i=1; i<=n; ++i) f *= i;
    return f;
}

__host__ double F_func(const double p, const double s) {
    return 2.0 * gamma0 * D * (cos(2.0 * p / 3.0) + 2.0 * cos(p / 3.0) * cos(M_PI * s / m));
}

__host__ double eps(const double p, const double s) {
    return gamma0 * sqrt(1.0 + 4.0 * cos(p) * cos(M_PI * s / m) + 4.0 * cos(M_PI * s / m) * cos(M_PI * s / m));
}

__host__ double eps_imp(const double p, const double s) {
    return 0.5 * (R + Q + sqrt((R - Q) * (R - Q) - 4.0 * (F_func(p, s) - eps(p, s) * eps(p, s) - D*D)));
}

__host__ double deltaUnderIntegral(double p, double alpha, double s) {
    return eps_imp(p, s) * cos(p * alpha);
}

__host__ double simpsonIntegralDelta(const double a, const double b, const int n, double alpha, double s) {
    const double width = (b-a)/n;
    double simpson_integral = 0;
    for(int step = 0; step < n; step++) {
        const double x1 = a + step*width;
        const double x2 = a + (step+1)*width;
        simpson_integral += (x2-x1)/6.0*(deltaUnderIntegral(x1, alpha, s) + 4.0*deltaUnderIntegral(0.5*(x1+x2), alpha, s) + deltaUnderIntegral(x2, alpha, s));
    }
    return simpson_integral;
}

__host__ double delta(const int alpha, const int s) {
    if (delta_alpha_s_cache.find({alpha, s}) != delta_alpha_s_cache.end()) {
        return delta_alpha_s_cache[{alpha, s}];
    }
    double result = simpsonIntegralDelta(-M_PI, M_PI, num_splits, alpha, s) / M_PI;
    delta_alpha_s_cache[{alpha, s}] = result;
    cout << "delta_alpha_s alpha = " << alpha << ", s = " << s << " cached" << endl;
    return result;
}

__host__ double GNominatorUnderIntegral(double p, double alpha, double s) {
    double sum = delta(0, s) / (2.0 * Kb * T);
    for (double alpha = 1.0; alpha <= ALPHA_MAX; alpha++) {
        sum += delta(alpha, s) * cos(alpha * p) / (Kb * T);
    }
    return cos(alpha * p) / (1.0 + exp(sum));
}

__host__ double simpsonIntegralGNominator(const double a, const double b, const int n, double alpha, double s) {
    const double width = (b-a)/n;
    double simpson_integral = 0;
    for(int step = 0; step < n; step++) {
        const double x1 = a + step*width;
        const double x2 = a + (step+1)*width;
        simpson_integral += (x2-x1)/6.0*(GNominatorUnderIntegral(x1, alpha, s) + 4.0*GNominatorUnderIntegral(0.5*(x1+x2), alpha, s) + GNominatorUnderIntegral(x2, alpha, s));
    }
    return simpson_integral;
}

__host__ double simpsonIntegralGDenominator(double p, double alpha, double s) {
    double sum = delta(0, s) / (2.0 * Kb * T);
    for (double alpha = 1.0; alpha <= ALPHA_MAX; alpha++) {
        sum += delta(alpha, s) * cos(alpha * p) / (Kb * T);
    }
    return 1.0 / ( 1.0 + exp(sum));
}

__host__ double simpsonIntegralGDenominator(const double a, const double b, const int n, double alpha, double s) {
    const double width = (b-a)/n;
    double simpson_integral = 0;
    for(int step = 0; step < n; step++) {
        const double x1 = a + step*width;
        const double x2 = a + (step+1)*width;
        simpson_integral += (x2-x1)/6.0*(simpsonIntegralGDenominator(x1, alpha, s) + 4.0*simpsonIntegralGDenominator(0.5*(x1+x2), alpha, s) + simpsonIntegralGDenominator(x2, alpha, s));
    }
    return simpson_integral;
}

__host__ double G(const int alpha) {
    if (G_alpha_cache.find(alpha) != G_alpha_cache.end()) {
        return G_alpha_cache[alpha];
    }
    double nominator = 0.0;
    double denominator = 0.0;
    for(double s = 1.0; s <= m; s++) {
        nominator += delta(alpha, s) / gamma0 * simpsonIntegralGNominator(-M_PI, M_PI, num_splits, alpha, s);
        denominator += simpsonIntegralGDenominator(-M_PI, M_PI, num_splits, alpha, s);
    }
    double result = -alpha * (nominator / denominator);
    G_alpha_cache[alpha] = result;
    cout << "G_alpha alpha = " << alpha << " cached" << endl;
    return result;
}

__host__ __device__ double fl(double l) {
    return pow(-1, l) / (factorial(l) * pow(2, 2 * l) * tgamma(l + 2));
}

/* Гауссово сглаживающее ядро SPH (1D).
 * Вход: расстояния r, длина сглаживания h, порядок производной
 */
__device__ double kernelDeriv0(double r, double h) {
    return 1.0 / h / SQRT_M_PI * exp(- r * r / (h * h));
}

__device__ double kernelDeriv1(double r, double h) {
    return pow(h, -3) / SQRT_M_PI * exp(- r * r / (h * h)) * (-2.0 * r);
}

__device__ double kernelDeriv2(double r, double h) {
    return pow(h, -5) / SQRT_M_PI * exp(-r * r / (h * h)) * (4.0 * r * r - 2.0 * h * h);
}

__device__ double kernelDeriv3(double r, double h) {
    return pow(h, -7) / sqrt(M_PI) * exp(-pow(r, 2) / pow(h, 2)) * (-8.0 * pow(r, 3) + 12.0 * pow(h, 2) * r);
}

__global__ void densityKernel(double* x, double* mass, double* h_array, Type* particles_type, double* rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles_type[i] != Type::FLEX) return;

    double sum = 0.0;
    double x_i = x[i];
    double uij = 0.0;
    double hij = 0.0;
    for (int j = 0; j < N; j++) {
        uij = x_i - x[j];
        hij = (h_array[i] + h_array[j]) / 2.0;
        sum += mass[j] * kernelDeriv0(uij, hij);
    }
    rho[i] = sum;
    __syncthreads();
    h_array[i] = H_COEF * mass[i] / rho[i];

}

/* Вычисление плотности в каждом из мест расположения частиц с помощью сглаживающего ядра
 * Входные данные: позиции частиц x, масса SPH-частицы m, длина сглаживания h
 */
__host__ void density(double* x, double* rho) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    densityKernel<<<gridSize, blockSize>>>(x_dev, mass_dev, h_array_dev, particles_type_dev, rho_dev);

    cudaMemcpy(rho, rho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void pressureKernelDRho(double* x, double* mass, double* h_array, Type* particles_type, double* drho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles_type[i] != Type::FLEX) return;

    double sum = 0.0;
    double x_i = x[i];
    double uij = 0.0;
    double hij = 0.0;
    for (int j = 0; j < N; j++) {
        uij = x_i - x[j];
        hij = (h_array[i] + h_array[j]) / 2.0;
        sum += mass[j] * kernelDeriv1(uij, hij);
    }
    drho[i] = sum;
}

__global__ void pressureKernelDDRho(double* x, double* mass, double* h_array, Type* particles_type, double* ddrho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles_type[i] != Type::FLEX) return;

    double sum = 0.0;
    double x_i = x[i];
    double uij = 0.0;
    double hij = 0.0;
    for (int j = 0; j < N; j++) {
        uij = x_i - x[j];
        hij = (h_array[i] + h_array[j]) / 2.0;
        sum += mass[j] * kernelDeriv2(uij, hij);
    }
    ddrho[i] = sum;
}

__global__ void pressureKernel(double* x, double* rho, double* drho, double* ddrho, double* mass, double* h_array, Type* particles_type, double* G_s_sum_array, double* P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles_type[i] != Type::FLEX) return;

    double sum = 0.0;
    double x_i = x[i];
    double uij = 0.0;
    double hij = 0.0;
    for (int j = 0; j < N; j++) {
        uij = x_i - x[j];
        hij = (h_array[i] + h_array[j]) / 2.0;
        sum += 0.25 * (drho[j] * drho[j] / rho[j] - ddrho[j]) / (chi * chi);
        sum = sum * mass[j] / rho[j] * kernelDeriv0(uij, hij);
    }
    P[i] = sum;
}

/* Вычисление давления на каждой из частиц с помощью сглаживающего ядра
 * P = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
 * Вход: положения x, плотности rho, масса SPH-частицы m, длина сглаживания h
 */
__host__ void pressure(double* x, double* rho, double* P) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_dev, rho, N * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    pressureKernelDRho<<<gridSize, blockSize>>>(x_dev, mass_dev, h_array_dev, particles_type_dev, drho_dev);
    pressureKernelDDRho<<<gridSize, blockSize>>>(x_dev, mass_dev, h_array_dev, particles_type_dev, ddrho_dev);

    cudaMemcpy(drho, drho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ddrho, ddrho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);

    pressureKernel<<<gridSize, blockSize>>>(x_dev, rho_dev, drho_dev, ddrho_dev, mass_dev, h_array_dev, particles_type_dev, G_s_sum_array_dev, P_dev);
    cudaMemcpy(P, P_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void accelerationKernel(double* x, double* u, double* rho, double* P, double b, double* mass, double* h_array, Type* particles_type, double* G_s_sum_array, double* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles_type[i] != Type::FLEX) return;

    double sum = 0.0;
    double x_i = x[i];
    double uij = 0.0;
    double hij = 0.0;

    // Дэмпирование и гармонический потенциал (0.5 x^2)
//    a[i] = - u[i] * b - x[i];

    double sum_nl = 0.0;
    for (int alpha = 1; alpha <= ALPHA_MAX; alpha++) {
        double l_sum = 0.0;
        for (int l = 0; l <= L_MAX; l++) {
            l_sum += fl(l) * l / (l + 1.0) * pow(alpha, 2 * l + 1) * pow(rho[i], l + 1);
        }
        sum_nl += alpha * G_s_sum_array[alpha - 1] * l_sum;
    }
//        printf("%lf = %lf\n", sum, sum_nl);
    double P_NL = 1.0 / (2.0 * chi * chi) * sum_nl;

    for (int j = 0; j < N; j++) {
        uij = x_i - x[j];
        hij = (h_array[i] + h_array[j]) / 2.0;
        sum += -mass[j] * (P_NL / pow(rho[i], 2) + P[j] / (rho[j] * rho[j]) /*- P_NL/(rho[i] * rho[i])*/) * kernelDeriv1(uij, hij);
    }
    a[i] = sum;
}

/* Расчёт ускорения каждой частицы под действием квантового давления, гармонического потенциала, демпфирования скорости
 * Входные данные: положения x, скорости u, масса SPH-частицы m, плотность rho, давление P, коэффициент демпфирования b
 */
__host__ void acceleration(double* x, double* u, double* rho, double* P, double b, double* a) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_dev, rho, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u_dev, u, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(P_dev, P, N * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    accelerationKernel<<<gridSize, blockSize>>>(x_dev, u_dev, rho_dev, P_dev, b, mass_dev, h_array_dev, particles_type_dev, G_s_sum_array_dev, a_dev);
    cudaMemcpy(a, a_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void probeDensityKernel(double* x, double* xx, double* mass, double* h_array, double* rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    double xx_i = xx[i];
    double uij = 0.0;
    double hij = 0.0;
    for (int j = 0; j < N; j++) {
        uij = xx_i - x[j];
        hij = (H_DEFAULT + h_array[j]) / 2.0;
        sum += mass[j] * kernelDeriv0(uij, hij);
    }
    rho[i] = sum;
}

/* Вычисление плотности в произвольных точках
 * Вход: положение x, масса частицы SPH m, масштабная длина h, точки измерения xx
 * Выход: плотность в равномерно расположенных точках
 */
__host__ void probeDensity(double* x, double* xx, double* prob_rho) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(xx_dev, xx, N * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    probeDensityKernel<<<gridSize, blockSize>>>(x_dev, xx_dev, mass_dev, h_array_dev, rho_dev);

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

    double v0 = (xStart + xEnd) / 2.0;
    for (int i = 0; i < N; i++) {
        if (i < SOLID_LAYER_LENGTH || i >= N - SOLID_LAYER_LENGTH) {
            rho[i] = a_eq;
            particles_type[i] = Type::SOLID;
        } else {
            rho[i] = a_eq * exp(-(x[i] - v0) * (x[i] - v0) / (b_eq * b_eq));
            particles_type[i] = Type::FLEX;
        }
    }
    cudaMemcpy(rho_dev, rho, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(particles_type_dev, particles_type, N * sizeof(Type), cudaMemcpyHostToDevice);

    // Инициализация масс частиц
    for (int i = 0; i < N; i++) {
        mass[i] = xStep * rho[i];//a_eq*a_eq * exp(-2.0 * (x[i] - v0) * (x[i] - v0) / (b_eq * b_eq));//xStep * rho[i];
    }
    cudaMemcpy(mass_dev, mass, N * sizeof(double), cudaMemcpyHostToDevice);

    // Инициализация сглаживающего расстояния
    for (int i = 0; i < N; i++) {
        h_array[i] = H_COEF * mass[i] / rho[i];
    }
    cudaMemcpy(h_array_dev, h_array, N * sizeof(double), cudaMemcpyHostToDevice);

    // Вычисление G_alpha
    double a0 = 0.0;
    double a1 = 0.0;
    for (int alpha = 1; alpha <= ALPHA_MAX; alpha++) {
        double G_alpha = G(alpha);
        G_s_sum_array[alpha - 1] = G_alpha;
        a0 += G_alpha * alpha;
        a1 -= G_alpha * alpha * alpha * alpha / 8.0;
    }
    cudaMemcpy(G_s_sum_array_dev, G_s_sum_array, ALPHA_MAX * sizeof(double), cudaMemcpyHostToDevice);
    cout << "SPH a0 = " << a0 << endl;
    cout << "SPH a1 = " << a1 << endl;


    // Инициализация плотности, давления и ускорения
    density(x, rho);
    pressure(x, rho, P);
    acceleration(x, u, rho, P, b, a);

    // v в t=-0.5*DT для leap frog интегратора
    for (int i = 0; i < N; i++) {
        u_mhalf[i] = u[i] - 0.5 * DT * a[i];
    }

    ofstream outfile("solution_cuda.txt");
    outfile << "X T Z" << endl;

    ofstream outfile_exact("solution_exact_cuda.txt");
    outfile_exact << "X T Z" << endl;

    // Главный цикл по времени
    double t = 0.0;
    for (int i = -NT_SETUP; i < NT; i++) {
        // Leap frog
        for (int j = 0; j < N; j++) {
            if (particles_type[j] != Type::FLEX) continue;
//            printf("%lf\n", a[j]);
            u_phalf[j] = u_mhalf[j] + a[j] * DT;
            x[j] = x[j] + u_phalf[j] * DT;
            u[j] = 0.5 * (u_mhalf[j] + u_phalf[j]);
            u_mhalf[j] = u_phalf[j];
        }

        if (i >= 0) {
            t = t + DT;
        }
        
        if ((i % PROGRESS_STEP) == 0) {
            int progress = (i / PROGRESS_STEP) + 1;
            cout << "SPH Progress: " << progress << "/" << N_PROGRESS << endl;
        }

        if (i == -1) {
            for (int j = 0; j < N; j++) {
                u_mhalf[j] = 1.0;
            }
            b = 0;
        }

        // Обновление плотностей, давлений, ускорений
        density(x, rho);
        pressure(x, rho, P);
        acceleration(x, u, rho, P, b, a);

        // Вывод в файлы
        if (i >= 0 && i % N_OUT == 0) {
            probeDensity(x, xx, probe_rho);
            for (int j = 0; j < N; j++) {
                outfile << xx[j] << " " << t << " " << probe_rho[j] / a_eq << endl; // TODO
//                outfile << x[j] << " " << t << " " << rho[j] / a_eq << endl;
            }
            for (int j = 0; j < N; j++) {
                double exact = 1.0 / sqrt(M_PI) * exp(-(xx[j] - sin(t)) * (xx[j]- sin(t)) / 2.0) * exp(-(xx[j] - sin(t)) * (xx[j]- sin(t)) / 2.0);
                outfile_exact << xx[j] << " " << t << " " << exact << endl;
            }
        }
    }
    outfile.close();
    outfile_exact.close();

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
