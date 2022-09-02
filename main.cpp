#include <iostream>
#include <vector>

#define M_PI 3.14159265358979323846


using namespace std;

class Besse {
public:
    int M; // Количество разбиений по x
    int N; // Количество разбиений по t
    double lambda;
    double A;
    double B;
    double C;
    
    double tStart = 0.0; // t начала
    double tStop = 5.0; // t конца
    
    vector<vector<double>> u; // Решение в точке [t_n][x_m]
    vector<double> x; // Сетка x
    vector<double> t; // Сетка t
    
    Besse(int M, int N) {
        this->M = M;
        this->N = N;
        prepareX(M);
        prepareT(N);
    }

    void prepareX(int M) {
        x = vector<double>(M + 2);
        const double h = 2 * M_PI / (M + 1);
        for (int m = 0; m <= M + 1; m++) {
            x[m] = -M_PI + m * h;
        }
    }
    
    void prepareT(int N) {
        t = vector<double>(N + 2);
        const double k = (tStop - tStart) / N;
        for (int n = 0; n <= N + 1; n++) {
            t[n] = n * k;
        }
    }

};

int main()
{
    Besse besse = Besse(20, 20);
}
