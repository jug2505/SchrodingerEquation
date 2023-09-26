#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>

using namespace std;

/* Гауссово сглаживающее ядро SPH (1D).
 * Вход: расстояние r, длина сглаживания h, порядок производной
 */
double kernel(double r, double h, int deriv) {
    double weight;
    switch (deriv) {
        case 0:
            weight = pow(h, -1) / sqrt(M_PI) * exp(-pow(r, 2) / pow(h, 2));
            break;
        case 1:
            weight = pow(h, -3) / sqrt(M_PI) * exp(-pow(r, 2) / pow(h, 2)) * (-2 * r);
            break;
        case 2:
            weight = pow(h, -5) / sqrt(M_PI) * exp(-pow(r, 2) / pow(h, 2)) * (4 * pow(r, 2) - 2 * pow(h, 2));
            break;
        case 3:
            weight = pow(h, -7) / sqrt(M_PI) * exp(-pow(r, 2) / pow(h, 2)) * (-8 * pow(r, 3) + 12 * pow(h, 2) * r);
            break;
        default:
            throw invalid_argument( "deriv not in [0, 1, 2, 3]" );
    }
    return weight;
}

/* Вычисление плотности в каждом из мест расположения частиц с помощью сглаживающего ядра
 * Входные данные: позиции частиц x, масса SPH-частицы m, длина сглаживания h
 */
vector<double> density(vector<double> x, double m, double h) {
    int n = (int) x.size();
    vector<double> rho(n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double uij = x[i] - x[j];
            rho[i] += m * kernel(uij, h, 0);
        }
    }
    return rho;
}

/* Вычисление давления на каждой из частиц с помощью сглаживающего ядра
 * P = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
 * Вход: положения x, плотности rho, масса SPH-частицы m, длина сглаживания h
 */
vector<double> pressure(vector<double> x, vector<double> rho, double m, double h) {
    int n = x.size();
    vector<double> P(n, 0.0);
    vector<double> drho(n, 0.0);
    vector<double> ddrho(n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double uij = x[i] - x[j];
            drho[i] += m * kernel(uij, h, 1);
            ddrho[i] += m * kernel(uij, h, 2);
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double uij = x[i] - x[j];
            double fac = -m * (P[i] / pow(rho[i], 2) + P[j] / pow(rho[j], 2));
            P[i] += fac * kernel(uij, h, 0);
        }
    }
    return P;
}

/* Расчёт ускорения каждой частицы под действием квантового давления, гармонического потенциала, демпфирования скорости
 * Входные данные: положения x, скорости u, масса SPH-частицы m, плотность rho, давление P, коэффициент демпфирования b
 */
vector<double> acceleration(vector<double> x, vector<double> u, double m, vector<double> rho, vector<double> P, double b, double h) {
    int n = x.size();
    vector<double> a(n, 0.0);
    for (int i = 0; i < n; i++) {
        // Дэмпирование и гармонический потенциал (0.5 x^2)
        a[i] -= u[i] * b + x[i];

        // Квантовое давление
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double uij = x[i] - x[j];
                double fac = -m * (P[i] / pow(rho[i], 2) + P[j] / pow(rho[j], 2));
                a[i] += fac * kernel(uij, h, 1);
            }
        }
    }
    return a;
}

/* Вычисление плотности в произвольных точках
 * Вход: положение x, масса частицы SPH m, масштабная длина h, точки измерения xx
 * Выход: плотность в равномерно расположенных точках
 */
vector<double> probeDensity(vector<double> x, double m, double h, vector<double> xx) {
    int n = (int) x.size();
    int nxx = (int) xx.size();
    vector<double> rr(nxx, 0.0);
    for (int i = 0; i < nxx; i++) {
        for (int j = 0; j < n; j++) {
            double uij = xx[i] - x[j];
            rr[i] += m * kernel(uij, h, 0);
        }
    }
    return rr;
}

int main() {
    auto begin = chrono::steady_clock::now();
    // i d_t psi + nabla^2/2 psi -x^2 psi/2 = 0
    // Потенциал: 1/2 x^2
    int n = 500;  // Кол-во частиц
    double dt = 0.02;  // Шаг по времени
    int nt = 100;  // Кол-во шагов по времени
    int nt_setup = 400;  // Кол-во шагов на настройку
    int n_out = 25;  // Вывод каждые n_out шагов
    double b = 4;  // Демпфирование скорости для настройки начального состояния
    double m = 1.0 / n;  // Масса частицы SPH ( m * n = 1 normalizes |wavefunction|^2 to 1)
    double h = 40.0 / n;  // Расстояние сглаживания
    double t = 0.0;

    // Инициализация положений и скоростей частиц
    vector<double> x(n);
    vector<double> u(n, 0.0);
    for (int i = 0; i < n; i++) {
        x[i] = -3.0 + i * (6.0 / (n - 1));
    }

    // Инициализация плотности, давления и ускорения
    vector<double> rho = density(x, m, h);
    vector<double> P = pressure(x, rho, m, h);
    vector<double> a = acceleration(x, u, m, rho, P, b, h);

    // v в t=-0.5*dt для leap frog интегратора
    vector<double> u_mhalf(n);
    for (int i = 0; i < n; i++) {
        u_mhalf[i] = u[i] - 0.5 * dt * a[i];
    }

    // Главный цикл по времени
    for (int i = -nt_setup; i < nt; i++) {
        // Leap frog
        vector<double> u_phalf(n);
        for (int j = 0; j < n; j++) {
            u_phalf[j] = u_mhalf[j] + a[j] * dt;
            x[j] += u_phalf[j] * dt;
            u[j] = 0.5 * (u_mhalf[j] + u_phalf[j]);
            u_mhalf[j] = u_phalf[j];
        }
        if (i >= 0) {
            t += dt;
        }
        cout << t << endl;

        if (i == -1) {  // Выключение демпфирования перед t=0
            for (int j = 0; j < n; j++) {
                u[j] = 1.0;
                u_mhalf[j] = u[j];
            }
            b = 0;
        }

        // Обновление плотностей, давлений, ускорений
        rho = density(x, m, h);
        P = pressure(x, rho, m, h);
        a = acceleration(x, u, m, rho, P, b, h);

        // Вывод в файлы
        if (i >= 0 && i % n_out == 0) {
            ofstream outfile("solution_t_" + to_string(t) + ".txt");

            outfile << "X Z" << endl;

            vector<double> xx(400);
            for (int j = 0; j < 400; j++) {
                xx[j] = -4.0 + j * (8.0 / 399);
            }
            vector<double> rr = probeDensity(x, m, h, xx);
            vector<double> rr_exact(400);
            for (int j = 0; j < 400; j++) {
                rr_exact[j] = 1.0 / sqrt(M_PI) * exp(-pow(xx[j] - sin(t), 2));
            }
            for (int j = 0; j < 400; j++) {
                outfile << xx[j] << " " << rr_exact[j] << endl;
            }
            outfile << endl;

            outfile.close();
        }
    }
    auto end = chrono::steady_clock::now();
    auto elapsed_m = std::chrono::duration_cast<chrono::seconds>(end - begin);
    cout << "Время работы: " << elapsed_m.count() << " секунд" << endl;

    return 0;
}

