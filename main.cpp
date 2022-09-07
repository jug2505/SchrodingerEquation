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

    MatrixXcd u; // Решение в точке [t_n][x_m]

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

};

int main()
{
    Besse besse = Besse(20, 20);
}
