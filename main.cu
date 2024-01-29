#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>

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
#define N 500
#define CENT_COEF 5
constexpr double DT = 0.002;  // Шаг по времени
constexpr int NT = 15000;  // Кол-во шагов по времени
constexpr int NT_SETUP = 0;  // Кол-во шагов на настройку
constexpr int N_OUT = 15;  // Вывод каждые N_OUT шагов

double b = 0;  // Демпфирование скорости для настройки начального состояния
// #define M (1.0 / N) // Масса частицы SPH ( M * n = 1 normalizes |wavefunction|^2 to 1)
#define h (40.0 * CENT_COEF / N)  // Расстояние сглаживания
constexpr double xStart = -10.0;
constexpr double xEnd = 10.0;
constexpr double xStep = (xEnd - xStart) / (N - 1);

// Настройка CUDA
#define BLOCK_SIZE 32
#define GRID_SIZE ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)

// Коэффициенты задачи
#define m 7
#define gamma0 4.32e-12
#define chi 8// 0: 20.0, 1: 40, 2: 8
#define E0 0.0
//#define omega 5.0e14
//#define omega0 1.0e14
#define Kb 1.38e-16
#define T 77.0
#define a_eq (0.00016*0.00016) // 0: 0.3, 1: (0.0323*0.0323), 2: (0.00016*0.00016)
#define b_eq 2.0
#define F 0
#define F0 1.0
#define S_MAX 7
#define ALPHA_MAX 9

// Кол-во разбиений для интеграла
//const int num_splits = 100000;
const int num_splits = 10000;

// Кэш
map<pair<int, int>, double> G_alpha_s_cache;
map<pair<int, int>, double> delta_alpha_s_cache;

// Данные на GPU
double *x_dev, *xx_dev, *rho_dev, *drho_dev, *ddrho_dev, *P_dev, *u_dev, *a_dev, *mass_dev, *G_s_sum_array_dev, *P_NL_dev;
// Данные на CPU
double *x, *u, *rho, *drho, *ddrho, *P, *a, *xx, *probe_rho, *u_mhalf, *u_phalf, *mass, *G_s_sum_array, *test_init_rho, *P_NL;

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
    G_s_sum_array = new double[ALPHA_MAX];
    test_init_rho = new double[N];
    P_NL = new double[N];

    cudaMalloc(&x_dev, N * sizeof(double));
    cudaMalloc(&xx_dev, N * sizeof(double));
    cudaMalloc(&rho_dev, N * sizeof(double));
    cudaMalloc(&drho_dev, N * sizeof(double));
    cudaMalloc(&ddrho_dev, N * sizeof(double));
    cudaMalloc(&P_dev, N * sizeof(double));
    cudaMalloc(&u_dev, N * sizeof(double));
    cudaMalloc(&a_dev, N * sizeof(double));
    cudaMalloc(&mass_dev, N * sizeof(double));
    cudaMalloc(&G_s_sum_array_dev, ALPHA_MAX * sizeof(double));
    cudaMalloc(&P_NL_dev, N * sizeof(double));
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
    delete[] G_s_sum_array;
    delete[] test_init_rho;
    delete[] P_NL;

    cudaFree(x_dev);
    cudaFree(xx_dev);
    cudaFree(rho_dev);
    cudaFree(drho_dev);
    cudaFree(ddrho_dev);
    cudaFree(P_dev);
    cudaFree(u_dev);
    cudaFree(a_dev);
    cudaFree(mass_dev);
    cudaFree(G_s_sum_array_dev);
    cudaFree(P_NL_dev);
    checkCudaErrors(cudaGetLastError());
}

__host__ __device__ double factorial(const int n) {
    double f = 1;
    for (int i=1; i<=n; ++i) f *= i;
    return f;
}

__host__ double eps(const double p, const double s) {
    return gamma0 * sqrt(1.0 + 4.0 * cos(p) * cos(M_PI * (s + F/F0) / m) + 4.0 * cos(M_PI * (s + F/F0) / m) * cos(M_PI * (s + F/F0) / m));
}

__host__ double deltaUnderIntegral(double p, double alpha, double s) {
    return eps(p, s) * cos(p * alpha);
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

__host__ double GNominatorUnderIntegral(double r, double alpha, double s) {
    double sum = 0.0;
    for (int i = 1; i <= 9; i++) {
        sum += delta(i, s) * cos(i * r) / (Kb * T);
    }
    return cos(alpha * r) * exp(-sum);
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

__host__ double simpsonIntegralGDenominator(double r, double alpha, double s_idx) {
    double sum = 0.0;
    for (int i = 1; i <= 9; i++) {
        sum += delta(i, s_idx) * cos(i * r) / (Kb * T);
    }
    return exp(-sum);
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

__host__ double G(const int alpha, const int s) {
    if (G_alpha_s_cache.find({alpha, s}) != G_alpha_s_cache.end()) {
        return G_alpha_s_cache[{alpha, s}];
    }

    double nominator = simpsonIntegralGNominator(-M_PI, M_PI, num_splits, alpha, s);

    double denominator = 0.0;
    for (int s_idx = 1; s_idx <= 7; s_idx++) {
        denominator += simpsonIntegralGDenominator(-M_PI, M_PI, num_splits, alpha, s_idx);
    }
    double result = -alpha * delta(alpha, s) * nominator / (gamma0 * denominator);
    G_alpha_s_cache[{alpha, s}] = result;
    cout << "G_alpha_s alpha = " << alpha << ", s = " << s << " cached" << endl;

    return result;
}

__host__ double G1(const int alpha, const int s, const double t_value) {
    return G(alpha, s) * cos(alpha * E0 * t_value * 0.0001);
}

__host__ double G2(const int alpha, const int s, const double t_value) {
    return G(alpha, s) * sin(alpha * E0 * t_value * 0.0001);
}

__host__ __device__ double fl(double l) {
    return pow(-1, l) / (factorial(l) * pow(2, 2 * l) * tgamma(l + 2));
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

__global__ void densityKernel(double* x, double* mass, double* rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        sum += mass[j] * kernelDeriv0(uij);
    }
    rho[i] = sum;
}

/* Вычисление плотности в каждом из мест расположения частиц с помощью сглаживающего ядра */
__host__ void density(double* x, double* rho) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);

    densityKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, rho_dev);

    cudaMemcpy(rho, rho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void pressureKernelDRho(double* x, double* mass, double* drho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        sum += mass[j] * kernelDeriv1(uij);
    }
    drho[i] = sum;
}

__global__ void pressureKernelDDRho(double* x, double* mass, double* ddrho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        sum += mass[j] * kernelDeriv2(uij);
    }
    ddrho[i] = sum;
}

__global__ void pressureKernel(double* x, double* mass, double* G_s_sum_array, double* rho, double* drho, double* ddrho, double* P, double* P_NL) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;

    // Обычное давление
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        sum += 0.25 * (drho[j] * drho[j] / rho[j] - ddrho[j]) * mass[j] / rho[j] * kernelDeriv0(uij) / (chi * chi);
    }
    P[i] = sum;

    // Давление от нелинейности
    double sum_nl = 0.0;
    for (int alpha = 1; alpha <= 9; alpha++) {
        double l_sum = 0.0;
        for (int l = 1; l <= 5; l++) {
            l_sum += fl(l) * l / (l + 1.0) * pow(alpha, 2 * l) * pow(rho[i], l + 1);
        }
        sum_nl += alpha * G_s_sum_array[alpha - 1] * l_sum;
    }
    P_NL[i] = - 1.0 / (2.0 * chi * chi) * sum_nl;
    //printf("%lf\n", P_NL[i]);
}

/* Вычисление давления на каждой из частиц с помощью сглаживающего ядра
 * P = -(1/(4*chi^2))*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
 */
__host__ void pressure(double* x, double* rho, double* P, double* P_NL) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_dev, rho, N * sizeof(double), cudaMemcpyHostToDevice);

    pressureKernelDRho<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, drho_dev);
    pressureKernelDDRho<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, ddrho_dev);

    cudaMemcpy(drho, drho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ddrho, ddrho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);

    pressureKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, G_s_sum_array_dev, rho_dev, drho_dev, ddrho_dev, P_dev, P_NL_dev);
    cudaMemcpy(P, P_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(P_NL, P_NL_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void accelerationKernel(double* x, double* mass, double* G_s_sum_array, double* u, double* rho, double* P, double* P_NL, double b, double* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Дэмпирование и гармонический потенциал (0.5 x^2)
    //a[i] = - u[i] * b - x[i];

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        sum += -mass[j] * (P_NL[i] / (rho[i] * rho[i]) + P[j] / (rho[j] * rho[j])) * kernelDeriv1(uij);
    }
    a[i] = sum;
}

/* Расчёт ускорения каждой частицы под действием квантового давления, гармонического потенциала, демпфирования скорости */
__host__ void acceleration(double* x, double* u, double* rho, double* P, double* P_NL, double b, double* a) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_dev, rho, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u_dev, u, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(P_dev, P, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(P_NL_dev, P_NL, N * sizeof(double), cudaMemcpyHostToDevice);

    accelerationKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, G_s_sum_array_dev, u_dev, rho_dev, P_dev, P_NL_dev, b, a_dev);

    cudaMemcpy(a, a_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void probeDensityKernel(double* x, double* mass, double* xx, double* rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = xx[i] - x[j];
        sum += mass[j] * kernelDeriv0(uij);
    }
    rho[i] = sum;
}

/* Вычисление плотности в произвольных точках */
__host__ void probeDensity(double* x, double* xx, double* prob_rho) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(xx_dev, xx, N * sizeof(double), cudaMemcpyHostToDevice);

    probeDensityKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, xx_dev, rho_dev);

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

    // Инициализация масс частиц
    double v0 = (xStart + xEnd) / 2.0;
    for (int i = 0; i < N; i++) {
        mass[i] = xStep * a_eq * exp(-(x[i] - v0) * (x[i] - v0) / (b_eq * b_eq)); // TODO
        // test_init_rho[i] = a_eq * exp(-(x[i] - v0) * (x[i] - v0) / (b_eq * b_eq));
    }
    cudaMemcpy(mass_dev, mass, N * sizeof(double), cudaMemcpyHostToDevice);

    // Вычисление G_alpha_s
    for (int alpha = 1; alpha <= ALPHA_MAX; alpha++) {
        double sum_s = 0.0;
        for(int s = 1; s <= S_MAX; s++) {
            sum_s += G(alpha, s);
        }
        G_s_sum_array[alpha - 1] = sum_s;
    }
    cudaMemcpy(G_s_sum_array_dev, G_s_sum_array, ALPHA_MAX * sizeof(double), cudaMemcpyHostToDevice);

    // Инициализация плотности, давления и ускорения
    density(x, rho);
    pressure(x, rho, P, P_NL);
    acceleration(x, u, rho, P, P_NL, b, a);

    // v в t=-0.5*DT для leap frog интегратора
    for (int i = 0; i < N; i++) {
        u_mhalf[i] = u[i] - 0.5 * DT * a[i];
    }

    ofstream outfile("solution_cuda.txt");
    // ofstream outfile_start("start_cuda.txt");
    outfile << "X T Z" << endl;
    // outfile_start << "X Z" << endl;


    // Главный цикл по времени
    double t = 0.0;
    for (int i = -NT_SETUP; i < NT; i++) {
        // Вывод в файлы
        if (i >= 0 && i % N_OUT == 0) {
            probeDensity(x, xx, probe_rho);
            for (int j = 0; j < N; j++) {
                outfile << xx[j] << " " << t << " " << probe_rho[j] / a_eq << endl; // TODO
                //outfile_start << xx[j] << " " << test_init_rho[j] << endl;
            }
        }

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
        cout << "SPH t steps: " << i << "/"<< NT << endl;

        if (i == -1) {
            for (int j = 0; j < N; j++) {
                u_mhalf[j] = 1.0;
            }
            b = 0.0;
        }

        // Обновление плотностей, давлений, ускорений
        density(x, rho);
        pressure(x, rho, P, P_NL);
        acceleration(x, u, rho, P, P_NL, b, a);
    }
    outfile.close();
    //outfile_start.close();

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
