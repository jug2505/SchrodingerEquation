#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

class Besse {
public:
    double lambda = 0.0; // TODO: С типом пока не понятно
    double A = 0.0; // TODO: С типом пока не понятно
    double B = 0.0; // TODO: С типом пока не понятно
    double C = 0.0; // TODO: С типом пока не понятно

    vector<vector<complex<double>>> u; // Решение в точке [t_n][x_m]

    vector<double> x; // Сетка x
    double h; // Шаг по x
    int M; // Количество разбиений по x

    vector<double> t; // Сетка t
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
        x = vector<double>(M + 2);
        h = 2 * M_PI / (M + 1);
        for (int m = 0; m <= M + 1; m++) {
            x[m] = -M_PI + m * h;
        }
    }

    void prepareT() {
        t = vector<double>(N + 1);
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
