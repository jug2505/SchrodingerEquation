#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

#include <Eigen/Core>

using namespace std;
using namespace Eigen;

class Besse {
public:
    double lambda = 0.0; // TODO: С типом пока не понятно
    double A = 0.0; // TODO: С типом пока не понятно
    double B = 0.0; // TODO: С типом пока не понятно
    double C = 0.0; // TODO: С типом пока не понятно

    MatrixXcd U; // Решение в точке [t_n][x_m]

    VectorXd x; // Сетка x
    double h; // Шаг по x
    int M; // Количество разбиений по x

    VectorXd t; // Сетка t
    double k; // Шаг по t
    int N; // Количество разбиений по t
    double tStart = 0.0; // t начала
    double tStop = 5.0; // t конца

    Besse(int M, int N) {
        this->M = M;
        this->N = N;

        prepareX();
        prepareT();
        prepareU();
    }

    void prepareX() {
        x = VectorXd(M + 2);
        h = 2 * M_PI / (M + 1);
        for (int m = 0; m <= M + 1; m++) {
            x[m] = -M_PI + m * h;
        }
    }

    void prepareT() {
        t = VectorXd(N + 1);
        k = (tStop - tStart) / N;
        for (int n = 0; n <= N; n++) {
            t[n] = n * k;
        }
    }

    void prepareU() {
        U = MatrixXcd(N + 1, M + 1);
        for (int m = 0; m <= M; m++) {
            U(0, m) = exp(complex<double>(0.0, 1.0) * x(m)); // e^(ix) - начальная функция
        }
    }

    void solve() {
        VectorXcd V_minus = VectorXcd(M + 1);
        VectorXcd V_plus = VectorXcd(M + 1);

        V_minus = U.row(0).cwiseAbs2(); // V^-1/2 = |U|^2

        complex<double> r = k / (2 * h * h);

        for (int n = 0; n < N; n++) { // 0 -> N-1

            VectorXcd tmp1 = -V_minus;
            VectorXcd tmp2 = 2.0 * U.row(n).cwiseAbs2();
            V_plus = tmp1 + tmp2;

            MatrixXcd A_plus = MatrixXcd(M + 1, M + 1);
            A_plus(0, 0) = complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(0);
            A_plus(0, 1) = 1.0;
            A_plus(0, M) = 1.0;
            A_plus(M, 0) = 1.0;
            A_plus(M, M - 1) = 1.0;
            A_plus(M, M) = complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(M);
            for (int m = 1; m < M; m++) {
                complex<double> a_plus = complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(m);
                A_plus(m,m - 1) = 1.0;
                A_plus(m, m) = a_plus;
                A_plus(m, m + 1) = 1.0;
            }
            A_plus = r * A_plus;

            MatrixXcd A_minus = MatrixXcd(M + 1, M + 1);
            A_minus(0, 0) = -complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(0);
            A_minus(0, 1) = 1.0;
            A_minus(0, M) = 1.0;
            A_minus(M, 0) = 1.0;
            A_minus(M, M - 1) = 1.0;
            A_minus(M, M) = -complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(M);
            for (int m = 1; m < M; m++) {
                complex<double> a_minus = -complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(m);
                A_minus(m,m - 1) = 1.0;
                A_minus(m, m) = a_minus;
                A_minus(m, m + 1) = 1.0;
            }
            A_minus = -r * A_minus;

            V_minus = V_plus;
        }
    }

};

int main()
{
    Besse besse = Besse(20, 20);
    besse.solve();
}
