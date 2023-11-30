#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

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

// Константы CUDA
#define BLOCK_SIZE 32

// Константы SPH
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

// На GPU
double* x_dev[2] = {NULL, NULL};
double* xx_dev[2] = {NULL, NULL};
double* rho_dev[2] = {NULL, NULL};
double* drho_dev[2] = {NULL, NULL};
double* ddrho_dev[2] = {NULL, NULL};
double* P_dev[2] = {NULL, NULL};
double* u_dev[2] = {NULL, NULL};
double* a_dev[2] = {NULL, NULL};

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

#define NUM_DEVICES 2

cudaDeviceProp prop; 
int it, nthr;
int deviceCount, deviceId[2], devId_t;
int can_access_peer;

void init() {
    omp_set_dynamic(0);
    omp_set_num_threads(NUM_DEVICES);

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

    cudaGetDeviceCount( &deviceCount );
	printf("deviceCount = %d\n", deviceCount);
	for (int i = 0; i < deviceCount; i++){
		cudaGetDeviceProperties(&prop, i);
		printf("Id_GPU = %d, Name_GPU = %s\n", i, prop.name);
	}
    printf("Input deviceId (0,1,...,%d)\n",deviceCount-1);
	for (int i = 0; i < 2; i++) {
		printf("Input deviceId = ");
		scanf("%d", &deviceId[i]);
		if (deviceId[i] > deviceCount - 1){
			printf("\n Net takogo nomera device GPU"); return 0;
		}
	}

    for (int i = 0; i < 2; i++) printf("deviceId[%d] = %d\n", i, deviceId[i]);

    #pragma omp parallel for schedule(static,1)
	for (int i = 0; i < 2; i++){
		cudaSetDevice(deviceId[i]);
		if (i == 0) devId_t = deviceId[i+1];
		else if (i == 1) devId_t = deviceId[i-1];
		cudaDeviceCanAccessPeer(&can_access_peer, deviceId[i], devId_t);
		printf("can_access_peer=%d, %d\n", can_access_peer, deviceId[i]);
	}

    #pragma omp parallel for num_threads(NUM_DEVICES)
    for(int i = 0; i < NUM_DEVICES; i++) {
        cudaSetDevice(i);
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

    #pragma omp parallel for num_threads(NUM_DEVICES)
    for(int i = 0; i < NUM_DEVICES; i++) {
        cudaSetDevice(i);
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
}

/* Гауссово сглаживающее ядро SPH (1D).
 * Вход: расстояния r, длина сглаживания h, порядок производной
 */
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

__global__ void densityKernel(double* x, double* rho, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double sum = 0.0;
    double x_i = x[i];
    double uij = 0.0;
    for (int j = 0; j < n; j++) {
        uij = x_i - x[j];
        sum += M * kernelDeriv0(uij);
    }
    rho[i] = sum;
}

/* Вычисление плотности в каждом из мест расположения частиц с помощью сглаживающего ядра
 * Входные данные: позиции частиц x, масса SPH-частицы m, длина сглаживания h
 */
__host__ void density(double* x, double* rho) {
    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    #pragma omp parallel for num_threads(NUM_DEVICES)
    for(int i = 0; i < NUM_DEVICES; i++) {
        cudaSetDevice(i);
        cudaMemcpy(x_dev + i*(N/NUM_DEVICES), x + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyHostToDevice);
        densityKernel<<<gridSize, blockSize>>>(x_dev + i*(N/NUM_DEVICES), rho_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES);
        cudaMemcpy(rho + i*(N/NUM_DEVICES), rho_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaGetLastError());
    }
}

__global__ void pressureKernelDRho(double* x, double* drho, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double sum = 0.0;
    double x_i = x[i];
    double uij = 0.0;
    for (int j = 0; j < n; j++) {
        uij = x_i - x[j];
        sum += M * kernelDeriv1(uij);
    }
    drho[i] = sum;
}

__global__ void pressureKernelDDRho(double* x, double* ddrho, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double sum = 0.0;
    double x_i = x[i];
    double uij = 0.0;
    for (int j = 0; j < n; j++) {
        uij = x_i - x[j];
        sum += M * kernelDeriv2(uij);
    }
    ddrho[i] = sum;
}

__global__ void pressureKernel(double* x, double* rho, double* drho, double* ddrho, double* P, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double sum = 0.0;
    double x_i = x[i];
    double uij = 0.0;
    for (int j = 0; j < n; j++) {
        uij = x_i - x[j];
        sum += 0.25 * (drho[j] * drho[j] / rho[j] - ddrho[j]) * M / rho[j] * kernelDeriv0(uij);
    }
    P[i] = sum;
}

/* Вычисление давления на каждой из частиц с помощью сглаживающего ядра
 * P = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
 * Вход: положения x, плотности rho, масса SPH-частицы m, длина сглаживания h
 */
__host__ void pressure(double* x, double* rho, double* P) {
    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    #pragma omp parallel for num_threads(NUM_DEVICES)
    for(int i = 0; i < NUM_DEVICES; i++) {
        cudaSetDevice(i);
        cudaMemcpy(x_dev + i*(N/NUM_DEVICES), x + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(rho_dev + i*(N/NUM_DEVICES), rho + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyHostToDevice);
        pressureKernelDRho<<<gridSize, blockSize>>>(x_dev + i*(N/NUM_DEVICES), drho_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES);
        pressureKernelDDRho<<<gridSize, blockSize>>>(x_dev + i*(N/NUM_DEVICES), ddrho_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES);
        cudaMemcpy(drho + i*(N/NUM_DEVICES), drho_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(ddrho + i*(N/NUM_DEVICES), ddrho_dev + i*(N/NUM_DEVICES) , N/NUM_DEVICES * sizeof(double), cudaMemcpyDeviceToHost);
        pressureKernel<<<gridSize, blockSize>>>(x_dev + i*(N/NUM_DEVICES), rho_dev + i*(N/NUM_DEVICES), drho_dev + i*(N/NUM_DEVICES), ddrho_dev + i*(N/NUM_DEVICES), P_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES);
        cudaMemcpy(P + i*(N/NUM_DEVICES), P_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaGetLastError());
    }
}

__global__ void accelerationKernel(double* x, double* u, double* rho, double* P, double b, double* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double sum = 0.0;
    double x_i = x[i];
    double uij = 0.0;

    // Дэмпирование и гармонический потенциал (0.5 x^2)
    a[i] = - u[i] * b - x[i];

    for (int j = 0; j < n; j++) {
        uij = x_i - x[j];
        sum += -M * (P[i] / pow(rho[i], 2) + P[j] / (rho[j] * rho[j])) * kernelDeriv1(uij);
    }
    a[i] = a[i] + sum;
}

/* Расчёт ускорения каждой частицы под действием квантового давления, гармонического потенциала, демпфирования скорости
 * Входные данные: положения x, скорости u, масса SPH-частицы m, плотность rho, давление P, коэффициент демпфирования b
 */
__host__ void acceleration(double* x, double* u, double* rho, double* P, double b, double* a) {
    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    #pragma omp parallel for num_threads(NUM_DEVICES)
    for(int i = 0; i < NUM_DEVICES; i++) {
        cudaSetDevice(i);
        cudaMemcpy(x_dev + i*(N/NUM_DEVICES), x + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(rho_dev + i*(N/NUM_DEVICES), rho + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(u_dev + i*(N/NUM_DEVICES), u + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(P_dev + i*(N/NUM_DEVICES), P + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyHostToDevice);

        accelerationKernel<<<gridSize, blockSize>>>(x_dev + i*(N/NUM_DEVICES), u_dev + i*(N/NUM_DEVICES), rho_dev + i*(N/NUM_DEVICES), P_dev + i*(N/NUM_DEVICES), b, a_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES);
        cudaMemcpy(a + i*(N/NUM_DEVICES), a_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaGetLastError());
    }
}

__global__ void probeDensityKernel(double* x, double* xx, double* rho, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double sum = 0.0;
    double xx_i = xx[i];
    double uij = 0.0;
    for (int j = 0; j < n; j++) {
        uij = xx_i - x[j];
        sum += M * kernelDeriv0(uij);
    }
    rho[i] = sum;
}

/* Вычисление плотности в произвольных точках
 * Вход: положение x, масса частицы SPH m, масштабная длина h, точки измерения xx
 * Выход: плотность в равномерно расположенных точках
 */
__host__ void probeDensity(double* x, double* xx, double* prob_rho) {
    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    #pragma omp parallel for num_threads(NUM_DEVICES)
    for(int i = 0; i < NUM_DEVICES; i++) {
        cudaSetDevice(i);
        cudaMemcpy(x_dev + i*(N/NUM_DEVICES), x + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(xx_dev + i*(N/NUM_DEVICES), xx + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyHostToDevice);
        probeDensityKernel<<<gridSize, blockSize>>>(x_dev + i*(N/NUM_DEVICES), xx_dev + i*(N/NUM_DEVICES), rho_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES);
        cudaMemcpy(prob_rho + i*(N/NUM_DEVICES), rho_dev + i*(N/NUM_DEVICES), N/NUM_DEVICES * sizeof(double), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaGetLastError());
    }
}

void getCudaInfo() {
    for(int i=0;i<NUM_DEVICES;i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cout << "Device id: " << i << endl;
        cout << "Device name: " << prop.name << endl;
    }
}


void compute() {
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

    ofstream outfile_exact("solution_exact_cuda.txt");
    outfile_exact << "X Z" << endl;

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
        //cout << "SPH: t = " << t << endl;

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
            probeDensity(x,xx, probe_rho);

//            VectorXd rr_exact = VectorXd::Zero((N));
//            for (int j = 0; j < N; j++) {
//                rr_exact(j) = 1.0 / sqrt(M_PI) * exp(-(xx(j) - sin(t)) * (xx(j)- sin(t)) / 2.0) * exp(-(xx(j) - sin(t)) * (xx(j)- sin(t)) / 2.0);
//            }

            for (int j = 0; j < N; j++) {
                outfile << xx[j] << " " << probe_rho[j] << endl;
            }
//            for (int j = 0; j < N; j++) {
//                outfile_exact << xx(j) << " " << rr_exact(j) << endl;
//            }
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
