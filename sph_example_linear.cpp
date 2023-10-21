#include <iostream>
#include <fstream>

#include <Eigen/Core>

using namespace std;
using namespace Eigen;


/* Гауссово сглаживающее ядро SPH (1D).
 * Вход: расстояния r, длина сглаживания h, порядок производной
 */
VectorXd kernel(VectorXd r, double h, int deriv) {
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
        default:
            throw invalid_argument( "deriv not in [0, 1, 2, 3]" );
    }
    return weights;
}

/* Вычисление плотности в каждом из мест расположения частиц с помощью сглаживающего ядра
 * Входные данные: позиции частиц x, масса SPH-частицы m, длина сглаживания h
 */
VectorXd density(VectorXd x, double m, double h) {
    int n = x.size();
    VectorXd rho = VectorXd::Zero(n);
    for (int i = 0; i < n; i++) {
        VectorXd uij = x(i) - x.array();
        VectorXd rho_ij = m * kernel(uij, h, 0);
        rho(i) = rho(i) + rho_ij.sum();
    }
    return rho;
}

/* Вычисление давления на каждой из частиц с помощью сглаживающего ядра
 * P = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
 * Вход: положения x, плотности rho, масса SPH-частицы m, длина сглаживания h
 */
VectorXd pressure(VectorXd x, VectorXd rho, double m, double h) {
    int n = x.size();
    VectorXd drho = VectorXd::Zero(n);
    VectorXd ddrho = VectorXd::Zero(n);
    VectorXd P = VectorXd::Zero(n);

    for (int i = 0; i < n; i++) {
        VectorXd uij = x(i) - x.array();
        VectorXd drho_ij = m * kernel(uij, h, 1);
        VectorXd ddrho_ij = m * kernel(uij, h, 2);
        drho(i) = drho_ij.sum();
        ddrho(i) = ddrho_ij.sum();
    }

    for (int i = 0; i < n; i++) {
        VectorXd uij = x(i) - x.array();
        VectorXd P_ij = 0.25 * (drho.array() * drho.array() / rho.array() - ddrho.array()) * m / rho.array() * kernel(uij, h, 0).array();
        P(i) = P_ij.sum();
    }
    return P;
}

VectorXd deleteElement(const VectorXd& vector, int indexToDelete) {
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

/* Расчёт ускорения каждой частицы под действием квантового давления, гармонического потенциала, демпфирования скорости
 * Входные данные: положения x, скорости u, масса SPH-частицы m, плотность rho, давление P, коэффициент демпфирования b
 */
VectorXd acceleration(VectorXd x, VectorXd u, double m, VectorXd rho, VectorXd P, double b, double h) {
    int n = x.size();
    VectorXd a = VectorXd::Zero(n);

    for (int i = 0; i < n; i++) {
        // Дэмпирование и гармонический потенциал (0.5 x^2)
        a(i) = a(i) - u(i) * b - x(i);

        // Квантовое давление
        VectorXd x_js = deleteElement(x, i);
        VectorXd P_js = deleteElement(P, i);
        VectorXd rho_js = deleteElement(rho, i);

        VectorXd uij = x(i) - x_js.array();

        VectorXd fac = -m * (P(i) / pow(rho(i), 2) + P_js.array() / (rho_js.array() * rho_js.array()));
        VectorXd pressure_a = fac.array() * kernel(uij, h, 1).array();

        a(i) = a(i) + pressure_a.sum();
    }

    return a;
}

/* Вычисление плотности в произвольных точках
 * Вход: положение x, масса частицы SPH m, масштабная длина h, точки измерения xx
 * Выход: плотность в равномерно расположенных точках
 */
VectorXd probeDensity(VectorXd x, double m, double h, VectorXd xx) {
    int nxx = xx.size();
    VectorXd rr = VectorXd::Zero(nxx);

    for (int i = 0; i < nxx; i++) {
        VectorXd uij = xx(i) - x.array();
        VectorXd rho_ij = m * kernel(uij, h, 0);
        rr(i) = rr(i) + rho_ij.sum();
    }

    return rr;
}


int main() {
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

    ofstream outfile("solution.txt");
    outfile << "X Z" << endl;

    ofstream outfile_exact("solution_exact.txt");
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
        cout << "SPH: t = " << t << endl;

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


    return 0;
}