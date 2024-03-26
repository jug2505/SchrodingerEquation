#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nlohmann/json.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>

using namespace std;
using json = nlohmann::json;

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

enum class Type{ FLEX, SOLID };

#define SQRT_M_PI 1.77245385091

// Настройка SPH
int N = 68;
int LAYER_LENGTH = 0;
double H_DEFAULT = 0.4;
double DT = 0.02;  // Шаг по времени
int NT = 100;  // Кол-во шагов по времени
int NT_SETUP = 400;  // Кол-во шагов на настройку
int N_OUT = 1;  // Вывод каждые N_OUT шагов // 1000 выводов
N_PROGRESS = 10;
PROGRESS_STEP = NT / N_PROGRESS;

double b = 4;  // Демпфирование скорости для настройки начального состояния
double xStart = -3.0;
double xEnd = 3.0;
double xStep = (xEnd - xStart) / (N - 1);

// Настройка CUDA
#define BLOCK_SIZE 32
#define GRID_SIZE ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)

// Коэффициенты задачи
double gamma0 = 4.32e-12;
double Kb = 1.38e-16;
double chi = 5.6; // 0: 20.0, 1: 40, 2: 8, 3: 4, 4: 5.6
double a_eq = (0.323*0.323);  // 0: 0.3, 1: (0.0323*0.0323), 2: (0.00016*0.00016), 3: 1, 4:(0.323*0.323) (0.0065*0.0065)
double b_eq = 2.0;
int m = 7;
int ALPHA_MAX = 10;
int L_MAX = 5;
double R = 0.0; // R = Q = -D = 0 , (-0.25*gamma0), (-0.5*gamma0)
double Q = R;
double D = -Q;

// Кол-во разбиений для интеграла
const int num_splits = 100000;

// Кэш
map<pair<int, int>, double> G_alpha_s_cache;
map<pair<int, int>, double> delta_alpha_s_cache;
map<int, double> G_alpha_cache;

// Данные на GPU
double *x_dev, *xx_dev, *rho_dev, *drho_dev, *ddrho_dev, *P_dev, *u_dev, *a_dev, *mass_dev, *G_s_sum_array_dev, *P_NL_dev, *h_dev;
Type* particles_type_dev;
// Данные на CPU
double *x, *u, *rho, *drho, *ddrho, *P, *a, *xx, *probe_rho, *u_mhalf, *u_phalf, *mass, *G_s_sum_array, *test_init_rho, *P_NL, *h;
Type* particles_type;

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
    h = new double[N];
    particles_type = new Type[N];

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
    cudaMalloc(&h_dev, N * sizeof(double));
    cudaMalloc(&particles_type_dev, N * sizeof(Type));
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
    delete[] h;
    delete[] particles_type;

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
    cudaFree(h_dev);
    cudaFree(particles_type_dev);
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


//__device__ double kernelDeriv0(double r, double h) {
//    double e = abs(r) / h;
//    double w = 2.0 / (3.0 * h);
//    if (r >= 0.0 && r < 1.0) {
//        w = w * (1.0 - 1.5*e*e + 0.75*e*e*e);
//    } else if (r >= 1 && r < 2) {
//        w = w * (0.25 * (2.0 - e) * (2.0 - e) * (2.0 - e));
//    } else {
//        w = 0.0;
//    }
//    return w;
//}
//__device__ double kernelDeriv1(double r, double h) {
//    double e = abs(r) / h;
//    double w = 2.0 / (3.0 * h);
//    if (r >= 0.0 && r < 1.0) {
//        w = w * ( 2.25*e*e - 3.0*e);
//    } else if (r >= 1 && r < 2) {
//        w = w * (-0.75 * (2.0 - e) * (2.0 - e));
//    } else {
//        w = 0.0;
//    }
//    return w;
//}
//__device__ double kernelDeriv2(double r, double h) {
//    double e = abs(r) / h;
//    double w = 2.0 / (3.0 * h);
//
//    if (r >= 0.0 && r < 1.0) {
//        w = w * (4.5*e - 3.0);
//    } else if (r >= 1 && r < 2) {
//        w = w * (3.0 - 1.5 * e);
//    } else {
//        w = 0.0;
//    }
//    return w;
//}


/* Гауссово сглаживающее ядро SPH (1D) */
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

__global__ void densityKernel(double* x, double* mass, double* h, Type* particles_type, double* rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles_type[i] != Type::FLEX) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        double hij = (h[i] + h[j]) / 2.0;
        sum += mass[j] * kernelDeriv0(uij, hij);
    }
    rho[i] = sum;
    __syncthreads();
    h[i] = 1.3 * mass[i] / rho[i];
}

/* Вычисление плотности в каждом из мест расположения частиц с помощью сглаживающего ядра */
__host__ void density(double* x, double* rho) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);

    densityKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, h_dev, particles_type_dev, rho_dev);

    cudaMemcpy(rho, rho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void pressureKernelDRho(double* x, double* mass, double* h, Type* particles_type, double* drho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles_type[i] != Type::FLEX) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        double hij = (h[i] + h[j]) / 2.0;
        sum += mass[j] * kernelDeriv1(uij, hij);
    }
    drho[i] = sum;
}

__global__ void pressureKernelDDRho(double* x, double* mass, double* h, Type* particles_type, double* ddrho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles_type[i] != Type::FLEX) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        double hij = (h[i] + h[j]) / 2.0;
        sum += mass[j] * kernelDeriv2(uij, hij);
    }
    ddrho[i] = sum;
}

__global__ void pressureKernel(double* x, double* mass, double* G_s_sum_array, double* rho, double* drho, double* ddrho, double* P, double* h, Type* particles_type, double* P_NL) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles_type[i] != Type::FLEX) return;

    double sum = 0.0;

    // Обычное давление
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        double hij = (h[i] + h[j]) / 2.0;
        sum += 0.25 * (drho[j] * drho[j] / rho[j] - ddrho[j]) * mass[j] / rho[j] * kernelDeriv0(uij, hij) / (chi * chi);
    }
    P[i] = sum;

    // Давление от нелинейности
//    double sum_nl = 0.0;
//    for (int alpha = 1; alpha <= ALPHA_MAX; alpha++) {
//        double l_sum = 0.0;
//        for (int l = 0; l <= L_MAX; l++) {
//            l_sum += fl(l) * l / (l + 1.0) * pow(alpha, 2 * l + 1) * pow(rho[i], l + 1);
//        }
//        sum_nl += G_s_sum_array[alpha - 1] * l_sum;
//    }
//    P_NL[i] = - 1.0 / (2.0 * chi * chi) * sum_nl;
    P_NL[i] = 0.0;
    //printf("%lf\n", P_NL[i]);
}

/* Вычисление давления на каждой из частиц с помощью сглаживающего ядра
 * P = -(1/(4*chi^2))*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
 */
__host__ void pressure(double* x, double* rho, double* P, double* P_NL) {
    cudaMemcpy(x_dev, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_dev, rho, N * sizeof(double), cudaMemcpyHostToDevice);

    pressureKernelDRho<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, h_dev, particles_type_dev, drho_dev);
    pressureKernelDDRho<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, h_dev, particles_type_dev, ddrho_dev);

    cudaMemcpy(drho, drho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ddrho, ddrho_dev, N * sizeof(double), cudaMemcpyDeviceToHost);

    pressureKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, G_s_sum_array_dev, rho_dev, drho_dev, ddrho_dev, P_dev, h_dev, particles_type_dev, P_NL_dev);
    cudaMemcpy(P, P_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(P_NL, P_NL_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void accelerationKernel(double* x, double* mass, double* G_s_sum_array, double* u, double* rho, double* P, double* P_NL, double b, double* h, Type* particles_type, double* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles_type[i] != Type::FLEX) return;

    // Дэмпирование и гармонический потенциал (0.5 x^2)
    //a[i] = - u[i] * b - x[i];

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = x[i] - x[j];
        double hij = (h[i] + h[j]) / 2.0;
        sum += -mass[j] * (P_NL[i] / (rho[i] * rho[i]) + P[j] / (rho[j] * rho[j])) * kernelDeriv1(uij, hij);
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

    accelerationKernel<<<GRID_SIZE, BLOCK_SIZE>>>(x_dev, mass_dev, G_s_sum_array_dev, u_dev, rho_dev, P_dev, P_NL_dev, b, h_dev, particles_type_dev, a_dev);

    cudaMemcpy(a, a_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

__global__ void probeDensityKernel(double* x, double* mass, double* xx, double* rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        double uij = xx[i] - x[j];
        sum += mass[j] * kernelDeriv0(uij, H_DEFAULT);
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


void compute(string filename) {
    // Инициализация плотности, давления и ускорения
    density(x, rho);
    pressure(x, rho, P, P_NL);
    acceleration(x, u, rho, P, P_NL, b, a);

    // v в t=-0.5*DT для leap frog интегратора
    for (int i = 0; i < N; i++) {
        u_mhalf[i] = u[i] - 0.5 * DT * a[i];
    }

    ofstream outfile(filename);
    outfile << "X T Z" << endl;
    ofstream outfile_exact("exact_" + filename);
    outfile << "X T Z" << endl;

    // Главный цикл по времени
    double t = 0.0;
    for (int i = -NT_SETUP; i < NT; i++) {
        // Вывод в файлы
        if (i >= 0 && i % N_OUT == 0) {
            probeDensity(x, xx, probe_rho);
            // for (int j = 0; j < N; j++) {
            //     outfile << xx[j] << " " << t << " " << probe_rho[j] / a_eq << endl; // TODO
            // }
            for (int j = 0; j < N; j++) {
                outfile << xx[j] << " " << t << " " << probe_rho[j] << endl; // TODO
            }
            for (int j = 0; j < N; j++) {
                double exact = 1.0 / sqrt(M_PI) * exp(-(xx[j] - sin(t)) * (xx[j]- sin(t)) / 2.0) * exp(-(xx[j] - sin(t)) * (xx[j]- sin(t)) / 2.0);
                outfile_exact << xx[j] << " " << t << " " << exact << endl;
            }
            // for (int j = 0; j < N; j++) {
            //     outfile << x[j] << " " << t << " " << rho[j] / a_eq << endl; // TODO
            // }
        }

        // Leap frog
        for (int j = 0; j < N; j++) {
            if (particles_type[i] != Type::FLEX) continue;
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
            b = 0.0;
        }

        // Обновление плотностей, давлений, ускорений
        density(x, rho);
        pressure(x, rho, P, P_NL);
        acceleration(x, u, rho, P, P_NL, b, a);
    }
    outfile.close();
}

class Solver {
public:
    static void solve_beam_equation() {
        getCudaInfo();

        ifstream configFile("../beam_conf.json");
        json config = json::parse(configFile);
        config.at("N").get_to(N);
        config.at("SOLID_LAYER_LENGTH").get_to(SOLID_LAYER_LENGTH);
        config.at("DT").get_to(DT);
        config.at("NT").get_to(NT);
        config.at("NT_SETUP").get_to(NT_SETUP);
        config.at("N_OUT").get_to(N_OUT);
        config.at("N_PROGRESS").get_to(N_PROGRESS);
        config.at("b").get_to(b);
        config.at("xStart").get_to(xStart);
        config.at("xEnd").get_to(xEnd);
        config.at("gamma0").get_to(gamma0);
        config.at("Kb").get_to(Kb);
        config.at("chi").get_to(chi);
        config.at("a_eq").get_to(a_eq);
        config.at("b_eq").get_to(b_eq);
        config.at("m").get_to(m);
        config.at("ALPHA_MAX").get_to(ALPHA_MAX);
        config.at("L_MAX").get_to(L_MAX);
        config.at("R").get_to(R);
        config.at("Q").get_to(Q);
        config.at("D").get_to(D);

        init();

        // Инициализация положений частиц
        for (int i = 0; i < N; i++) {
            x[i] = xStart + i * xStep;
            xx[i] = x[i]; // Для графика
        }

        // Инициализация плотности, массы, сглаживающего расстояния
        double v0 = (xStart + xEnd) / 2.0;
        for (int i = 0; i < N; i++) {
            if (i < LAYER_LENGTH || i > N - LAYER_LENGTH) {
                rho[i] = a_eq;
                particles_type[i] = Type::SOLID;
            } else {
                rho[i] = a_eq * exp(-(x[i] - v0) * (x[i] - v0) / (b_eq * b_eq));
                particles_type[i] = Type::FLEX;
            }
            mass[i] = xStep * rho[i];
            h[i] = 1.3 * mass[i] / rho[i];
        }
        cudaMemcpy(rho_dev, rho, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(particles_type_dev, particles_type, N * sizeof(Type), cudaMemcpyHostToDevice);
        cudaMemcpy(mass_dev, mass, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(h_dev, h, N * sizeof(double), cudaMemcpyHostToDevice);

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
        
        compute("solution_cuda_beam.txt");
        
        clear();
    }

    static void solve_test_schrodinger() {
        getCudaInfo();

        ifstream configFile("../beam_conf.json");
        json config = json::parse(configFile);
        config.at("N").get_to(N);
        config.at("SOLID_LAYER_LENGTH").get_to(SOLID_LAYER_LENGTH);
        config.at("DT").get_to(DT);
        config.at("NT").get_to(NT);
        config.at("NT_SETUP").get_to(NT_SETUP);
        config.at("N_OUT").get_to(N_OUT);
        config.at("N_PROGRESS").get_to(N_PROGRESS);
        config.at("b").get_to(b);
        config.at("xStart").get_to(xStart);
        config.at("xEnd").get_to(xEnd);
        config.at("gamma0").get_to(gamma0);
        config.at("Kb").get_to(Kb);
        config.at("chi").get_to(chi);
        config.at("a_eq").get_to(a_eq);
        config.at("b_eq").get_to(b_eq);
        config.at("m").get_to(m);
        config.at("ALPHA_MAX").get_to(ALPHA_MAX);
        config.at("L_MAX").get_to(L_MAX);
        config.at("R").get_to(R);
        config.at("Q").get_to(Q);
        config.at("D").get_to(D);

        init();

        // Инициализация положений частиц
        for (int i = 0; i < N; i++) {
            x[i] = xStart + i * xStep;
            xx[i] = x[i]; // Для графика
        }

        // Инициализация плотности, массы, сглаживающего расстояния
        double v0 = (xStart + xEnd) / 2.0;
        for (int i = 0; i < N; i++) {
            if (i < LAYER_LENGTH || i > N - LAYER_LENGTH) {
                particles_type[i] = Type::SOLID;
            } else {
                particles_type[i] = Type::FLEX;
            }
            mass[i] = 1.0 / N;
            h[i] = 0.4;
        }
        cudaMemcpy(particles_type_dev, particles_type, N * sizeof(Type), cudaMemcpyHostToDevice);
        cudaMemcpy(mass_dev, mass, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(h_dev, h, N * sizeof(double), cudaMemcpyHostToDevice);

        compute("solution_cuda_test_schrodinger.txt");
        
        clear();
    }
}


int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    Solver::solve_test_schrodinger();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Время работы: %3.1f s\n", elapsedTime / 1000.0);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
