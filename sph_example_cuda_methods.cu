#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

#include <Eigen/Core>

using namespace std;
using namespace Eigen;

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

#define BLOCK_SIZE 1024

/* Гауссово сглаживающее ядро SPH (1D).
 * Вход: расстояния r, длина сглаживания h, порядок производной
 */
__host__ __device__ VectorXd kernel(VectorXd r, double h, int deriv) {
    int n = r.size();
    VectorXd weights = VectorXd::Zero(n);

    switch (deriv) {
        case 0:
            for (int i = 0; i < n; i++) {
                weights(i) = pow(h, -1) / sqrt(M_PI) * exp(-pow(r(i), 2) / pow(h, 2));
            }
            break;
        case 1:
            for (int i = 0; i < n; i++) {
                weights(i) = pow(h, -3) / sqrt(M_PI) * exp(-pow(r(i), 2) / pow(h, 2)) * (-2.0 * r(i));
            }
            break;
        case 2:
            for (int i = 0; i < n; i++) {
                weights(i) = pow(h, -5) / sqrt(M_PI) * exp(-pow(r(i), 2) / pow(h, 2)) * (4.0 * pow(r(i), 2) - 2.0 * pow(h, 2));
            }
            break;
        case 3:
            for (int i = 0; i < n; i++) {
                weights(i) = pow(h, -7) / sqrt(M_PI) * exp(-pow(r(i), 2) / pow(h, 2)) * (-8.0 * pow(r(i), 3) + 12.0 * pow(h, 2) * r(i));
            }
            break;
    }
    return weights;
}

__host__ __device__ VectorXd kernelDeriv0(VectorXd r, double h) {
    int n = r.size();
    VectorXd weights = VectorXd::Zero(n);

    for (int i = 0; i < n; i++) {
        weights(i) = pow(h, -1) / sqrt(M_PI) * exp(-pow(r(i), 2) / pow(h, 2));
    }

    return weights;
}

__host__ __device__ VectorXd kernelDeriv1(VectorXd r, double h) {
    int n = r.size();
    VectorXd weights = VectorXd::Zero(n);

    for (int i = 0; i < n; i++) {
        weights(i) = pow(h, -3) / sqrt(M_PI) * exp(-pow(r(i), 2) / pow(h, 2)) * (-2.0 * r(i));
    }

    return weights;
}

__host__ __device__ VectorXd kernelDeriv2(VectorXd r, double h) {
    int n = r.size();
    VectorXd weights = VectorXd::Zero(n);

    for (int i = 0; i < n; i++) {
        weights(i) = pow(h, -5) / sqrt(M_PI) * exp(-pow(r(i), 2) / pow(h, 2)) * (4.0 * pow(r(i), 2) - 2.0 * pow(h, 2));
    }

    return weights;
}

__host__ __device__ VectorXd kernelDeriv3(VectorXd r, double h) {
    int n = r.size();
    VectorXd weights = VectorXd::Zero(n);

    for (int i = 0; i < n; i++) {
        weights(i) = pow(h, -7) / sqrt(M_PI) * exp(-pow(r(i), 2) / pow(h, 2)) * (-8.0 * pow(r(i), 3) + 12.0 * pow(h, 2) * r(i));
    }

    return weights;
}

__global__ void densityKernel(double* x, double* rho, double m, double h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    VectorXd xVec = Map<VectorXd>(x, n);
    VectorXd uij = xVec(i) - xVec.array();
    VectorXd rho_ij = m * kernelDeriv0(uij, h);
    rho[i] = rho_ij.sum();
}

/* Вычисление плотности в каждом из мест расположения частиц с помощью сглаживающего ядра
 * Входные данные: позиции частиц x, масса SPH-частицы m, длина сглаживания h
 */
__host__ VectorXd density(VectorXd x, double m, double h) {
    int n = x.size();
    VectorXd rho = VectorXd::Zero(n);

    double* x_dev;
    double* rho_dev;
    HANDLE_ERROR(cudaMalloc(&x_dev, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&rho_dev, n * sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(x_dev, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;

    densityKernel<<<gridSize, blockSize>>>(x_dev, rho_dev, m, h, n);

    HANDLE_ERROR(cudaMemcpy(rho.data(), rho_dev, n * sizeof(double), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(x_dev));
    HANDLE_ERROR(cudaFree(rho_dev));

    return rho;
}

__global__ void pressureKernelDRho(double* x, double* drho, double m, double h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    VectorXd xVec = Map<VectorXd>(x, n);
    VectorXd uij = xVec(i) - xVec.array();
    VectorXd drho_ij = m * kernelDeriv1(uij, h);
    drho[i] = drho_ij.sum();
}

__global__ void pressureKernelDDRho(double* x, double* ddrho, double m, double h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    VectorXd xVec = Map<VectorXd>(x, n);
    VectorXd uij = xVec(i) - xVec.array();
    VectorXd ddrho_ij = m * kernelDeriv2(uij, h);
    ddrho[i] = ddrho_ij.sum();
}

__global__ void pressureKernel(double* x, double* P, double* rho, double* drho, double* ddrho, double m, double h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    VectorXd xVec = Map<VectorXd>(x, n);
    VectorXd rhoVec = Map<VectorXd>(rho, n);
    VectorXd drhoVec = Map<VectorXd>(drho, n);
    VectorXd ddrhoVec = Map<VectorXd>(ddrho, n);

    VectorXd uij = xVec(i) - xVec.array();
    VectorXd P_ij = 0.25 * (drhoVec.array() * drhoVec.array() / rhoVec.array() - ddrhoVec.array()) * m / rhoVec.array() * kernelDeriv0(uij, h).array();
    P[i] = P_ij.sum();
}

/* Вычисление давления на каждой из частиц с помощью сглаживающего ядра
 * P = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
 * Вход: положения x, плотности rho, масса SPH-частицы m, длина сглаживания h
 */
__host__ VectorXd pressure(VectorXd x, VectorXd rho, double m, double h) {
    int n = x.size();
    VectorXd drho = VectorXd::Zero(n);
    VectorXd ddrho = VectorXd::Zero(n);
    VectorXd P = VectorXd::Zero(n);

    double* x_dev;
    double* rho_dev;
    double* drho_dev;
    double* ddrho_dev;
    double* P_dev;
    HANDLE_ERROR(cudaMalloc(&x_dev, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&rho_dev, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&drho_dev, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&ddrho_dev, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&P_dev, n * sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(x_dev, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(rho_dev, rho.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;

    pressureKernelDRho<<<gridSize, blockSize>>>(x_dev, drho_dev, m, h, n);
    HANDLE_ERROR(cudaMemcpy(drho.data(), drho_dev, n * sizeof(double), cudaMemcpyDeviceToHost));

    pressureKernelDDRho<<<gridSize, blockSize>>>(x_dev, ddrho_dev, m, h, n);
    HANDLE_ERROR(cudaMemcpy(ddrho.data(), ddrho_dev, n * sizeof(double), cudaMemcpyDeviceToHost));

    pressureKernel<<<gridSize, blockSize>>>(x_dev, P_dev, rho_dev, drho_dev, ddrho_dev, m, h, n);
    HANDLE_ERROR(cudaMemcpy(P.data(), P_dev, n * sizeof(double), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(x_dev));
    HANDLE_ERROR(cudaFree(rho_dev));
    HANDLE_ERROR(cudaFree(drho_dev));
    HANDLE_ERROR(cudaFree(ddrho_dev));
    HANDLE_ERROR(cudaFree(P_dev));

    return P;

//
//
//    /////////////////
//    int n = x.size();
//    VectorXd drho = VectorXd::Zero(n);
//    VectorXd ddrho = VectorXd::Zero(n);
//    VectorXd P = VectorXd::Zero(n);
//
//    #pragma omp parallel for
//    for (int i = 0; i < n; i++) {
//        VectorXd uij = x(i) - x.array();
//        VectorXd drho_ij = m * kernel(uij, h, 1);
//        VectorXd ddrho_ij = m * kernel(uij, h, 2);
//        drho(i) = drho_ij.sum();
//        ddrho(i) = ddrho_ij.sum();
//    }
//
//    #pragma omp parallel for
//    for (int i = 0; i < n; i++) {
//        VectorXd uij = x(i) - x.array();
//        VectorXd P_ij = 0.25 * (drho.array() * drho.array() / rho.array() - ddrho.array()) * m / rho.array() * kernel(uij, h, 0).array();
//        P(i) = P_ij.sum();
//    }
//    return P;
}

__host__ __device__ VectorXd deleteElement(const VectorXd& vector, int indexToDelete) {
    int size = vector.size();
    VectorXd result = VectorXd::Zero(size - 1);

    int resultIndex = 0;
    for (int i = 0; i < size; ++i) {
        if (i == indexToDelete) {
            continue;
        }
        result(resultIndex) = vector(i);
        ++resultIndex;
    }

    return result;
}

__global__ void accelerationKernel(double* x, double* u, double* rho, double* P, double m, double h, double b, int n, double* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    VectorXd xVec = Map<VectorXd>(x, n);
    VectorXd rhoVec = Map<VectorXd>(rho, n);
    VectorXd PVec = Map<VectorXd>(P, n);

    // Дэмпирование и гармонический потенциал (0.5 x^2)
    a[i] = - u[i] * b - x[i];

    // Квантовое давление
    VectorXd x_js = deleteElement(xVec, i);
    VectorXd P_js = deleteElement(PVec, i);
    VectorXd rho_js = deleteElement(rhoVec, i);

    VectorXd uij = xVec(i) - x_js.array();

    VectorXd fac = -m * (PVec(i) / pow(rhoVec(i), 2) + P_js.array() / (rho_js.array() * rho_js.array()));
    VectorXd pressure_a = fac.array() * kernelDeriv1(uij, h).array();

    a[i] = a[i] + pressure_a.sum();
}

/* Расчёт ускорения каждой частицы под действием квантового давления, гармонического потенциала, демпфирования скорости
 * Входные данные: положения x, скорости u, масса SPH-частицы m, плотность rho, давление P, коэффициент демпфирования b
 */
__host__ VectorXd acceleration(VectorXd x, VectorXd u, double m, VectorXd rho, VectorXd P, double b, double h) {
    int n = x.size();
    VectorXd a = VectorXd::Zero(n);

    double* x_dev;
    double* rho_dev;
    double* u_dev;
    double* a_dev;
    double* P_dev;
    HANDLE_ERROR(cudaMalloc(&x_dev, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&rho_dev, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&u_dev, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&a_dev, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&P_dev, n * sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(x_dev, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(rho_dev, rho.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(u_dev, u.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(P_dev, P.data(), n * sizeof(double), cudaMemcpyHostToDevice));


    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;

    accelerationKernel<<<gridSize, blockSize>>>(x_dev, u_dev, rho_dev, P_dev, m, h, b, n, a_dev);
    HANDLE_ERROR(cudaMemcpy(a.data(), a_dev, n * sizeof(double), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(x_dev));
    HANDLE_ERROR(cudaFree(rho_dev));
    HANDLE_ERROR(cudaFree(u_dev));
    HANDLE_ERROR(cudaFree(a_dev));
    HANDLE_ERROR(cudaFree(P_dev));

    return a;


//
//
//    ////
//    int n = x.size();
//    VectorXd a = VectorXd::Zero(n);
//
//    #pragma omp parallel for
//    for (int i = 0; i < n; i++) {
//        // Дэмпирование и гармонический потенциал (0.5 x^2)
//        a(i) = a(i) - u(i) * b - x(i);
//
//        // Квантовое давление
//        VectorXd x_js = deleteElement(x, i);
//        VectorXd P_js = deleteElement(P, i);
//        VectorXd rho_js = deleteElement(rho, i);
//
//        VectorXd uij = x(i) - x_js.array();
//
//        VectorXd fac = -m * (P(i) / pow(rho(i), 2) + P_js.array() / (rho_js.array() * rho_js.array()));
//        VectorXd pressure_a = fac.array() * kernel(uij, h, 1).array();
//
//        a(i) = a(i) + pressure_a.sum();
//    }
//
//    return a;
}

/* Вычисление плотности в произвольных точках
 * Вход: положение x, масса частицы SPH m, масштабная длина h, точки измерения xx
 * Выход: плотность в равномерно расположенных точках
 */
VectorXd probeDensity(VectorXd x, double m, double h, VectorXd xx) {
    int nxx = xx.size();
    VectorXd rr = VectorXd::Zero(nxx);

    #pragma omp parallel for
    for (int i = 0; i < nxx; i++) {
        VectorXd uij = xx(i) - x.array();
        VectorXd rho_ij = m * kernelDeriv0(uij, h);
        rr(i) = rr(i) + rho_ij.sum();
    }

    return rr;
}


int main() {
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

    omp_set_dynamic(0);
    omp_set_num_threads(4);
    auto begin = chrono::steady_clock::now();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    // i d_t psi + nabla^2/2 psi -x^2 psi/2 = 0
    // Потенциал: 1/2 x^2
    int n = 100;  // Кол-во частиц
    double dt = 0.02;  // Шаг по времени
    int nt = 100;  // Кол-во шагов по времени
    int nt_setup = 400;  // Кол-во шагов на настройку
    int n_out = 25;  // Вывод каждые n_out шагов
    double b = 4;  // Демпфирование скорости для настройки начального состояния
    double m = 1.0 / n;  // Масса частицы SPH ( m * n = 1 normalizes |wavefunction|^2 to 1)
    double h = 40.0 / n;  // Расстояние сглаживания
    double t = 0.0;

    double xStart = -3.0;
    double xEnd = 3.0;
    double xStep = (xEnd - xStart) / (n - 1);

    // Инициализация положений и скоростей частиц
    VectorXd x = VectorXd::Zero(n);
    for (int i = 0; i < n; i++) {
        x[i] = xStart + i * xStep;
    }

    VectorXd u = VectorXd::Zero(n);

    // Инициализация плотности, давления и ускорения
    VectorXd rho = density(x, m, h);
    VectorXd P = pressure(x, rho, m, h);
    VectorXd a = acceleration(x, u, m, rho, P, b, h);

    // v в t=-0.5*dt для leap frog интегратора
    VectorXd u_mhalf = u - 0.5 * dt * a;

    ofstream outfile("solution_cuda.txt");
    outfile << "X Z" << endl;

    ofstream outfile_exact("solution_exact_cuda.txt");
    outfile_exact << "X Z" << endl;

    // Главный цикл по времени
    for (int i = -nt_setup; i < nt; i++) {
        // Leap frog
        VectorXd u_phalf = u_mhalf + a * dt;
        x = x + u_phalf *dt;
        u = 0.5 * (u_mhalf + u_phalf);
        u_mhalf = u_phalf;
        if (i >= 0) {
            t = t + dt;
        }
        //cout << "SPH: t = " << t << endl;

        if (i == -1) {
            u = VectorXd::Ones(n);
            u_mhalf = u;
            b = 0;
        }

        // Обновление плотностей, давлений, ускорений
        rho = density(x, m, h);
        P = pressure(x, rho, m, h);
        a = acceleration(x, u, m, rho, P, b, h);


        // Вывод в файлы
        if (i >= 0 && i % n_out == 0) {

            VectorXd xx = VectorXd::Zero(n);
            for (int j = 0; j < n; j++) {
                xx[j] = xStart + j * xStep;
            }

            VectorXd rr = probeDensity(x, m, h, xx);


            VectorXd rr_exact = VectorXd::Zero((n));
            for (int j = 0; j < n; j++) {
                rr_exact(j) = 1.0 / sqrt(M_PI) * exp(-(xx(j) - sin(t)) * (xx(j)- sin(t)) / 2.0) * exp(-(xx(j) - sin(t)) * (xx(j)- sin(t)) / 2.0);
            }

            for (int j = 0; j < n; j++) {
                outfile << xx(j) << " " << rr(j) << endl;
            }
            for (int j = 0; j < n; j++) {
                outfile_exact << xx(j) << " " << rr_exact(j) << endl;
            }
        }
    }
    outfile.close();
    outfile_exact.close();

    auto end = chrono::steady_clock::now();
    auto elapsed_m = std::chrono::duration_cast<chrono::seconds>(end - begin);
    cout << "Время работы: " << elapsed_m.count() << " секунд" << endl;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Constant Memory: %3.1f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}