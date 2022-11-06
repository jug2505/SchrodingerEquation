#include <iostream>
#include <utility>
#include <vector>
#include <complex>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/LU>
#include <fstream>

using namespace std;
using namespace Eigen;

class Besse {
public:
    function<complex<double>(double)> init_func; // f(x) инициализации

    MatrixXcd U; // Решение в точке [t_n][x_m]
    MatrixXcd U_analytic;

    VectorXd x; // Сетка x
    double h = 0.0; // Шаг по x
    int M; // Количество разбиений по x

    VectorXd t; // Сетка t
    double k = 0.0; // Шаг по t
    int N; // Количество разбиений по t

    double lambda;
    double A, B, C;

    double t_start = 0.0; // t начала
    double t_stop = 1.0; // t конца

    Besse(int M, int N, double t_start, double t_stop, double lambda,
          function<complex<double>(double)> init_func, double A, double B, double C) {
        this->M = M;
        this->N = N;
        this->t_start = t_start;
        this->t_stop = t_stop;
        this->lambda = lambda;
        this->init_func = std::move(init_func);
        this->A = A;
        this->B = B;
        this->C = C;

        prepare_x();
        prepare_t();
        prepare_u();
    }

    void prepare_x() {
        cout << "BESSE: prepare X" << endl;
        x = VectorXd(M + 1); // TODO: M + 2?
        h = 2 * M_PI / (M + 1);
        for (int m = 0; m <= M; m++) { // TODO: M + 1?
            x[m] = -M_PI + m * h;
        }
        cout << "h: " << h << endl;
    }

    void prepare_t() {
        cout << "BESSE: prepare T" << endl;
        t = VectorXd(N + 1);
        k = (t_stop - t_start) / N;
        for (int n = 0; n <= N; n++) {
            t[n] = n * k;
        }
        cout << "k: " << k << endl;
    }

    void prepare_u() {
        cout << "BESSE: prepare U" << endl;
        U = MatrixXcd(N + 1, M + 1);
//        for (int m = 0; m <= M; m++) {
//            U(0, m) = init_func(x(m));
//
//        }
        // Bi-soliton
        for (int x_idx = 0; x_idx < x.size(); x_idx++) {
            complex<double> t1 = 2.0 * complex<double>(0.0, 1.0);
            complex<double> _tmp1 = (1.0 + 2.0 * complex<double>(0.0, 1.0) * t(0)) * cosh(x(x_idx)) - x(x_idx) * sinh(x(x_idx));
            complex<double> _tmp2 = 1.0 + 2.0 * x(x_idx) * x(x_idx) + 8.0 * t(0) * t(0) + cosh(2.0 * x(x_idx));
            complex<double> _tmp3 = 4.0 * exp(complex<double>(0.0, 1.0) * t(0));
            complex<double> _tmp4 = _tmp3 * _tmp1 / _tmp2;
            U(0, x_idx) = _tmp4;
        }
    }

    void write_real_matrix_to_file(const MatrixXcd& src, const string& path_and_name) {
        MatrixXd real = src.real();
        write_matrix_to_file(real, path_and_name);
    }

    void write_imag_matrix_to_file(const MatrixXcd& src, const string& path_and_name) {
        MatrixXd imag = src.imag();
        write_matrix_to_file(imag, path_and_name);
    }

    void write_abs_matrix_to_file(const MatrixXcd& src, const string& path_and_name) {
        MatrixXd imag = MatrixXd(src.rows(), src.cols());
        for (int t_idx = 0; t_idx < src.rows(); t_idx++) {
            for (int x_idx = 0; x_idx < src.cols(); x_idx++) {
               imag(t_idx, x_idx) = abs(src(t_idx, x_idx));
            }
        }
        write_matrix_to_file(imag, path_and_name);
    }

    void write_matrix_to_file(const MatrixXd& src, const string& path_and_name) {
        ofstream file(path_and_name, ios::out | ios::trunc);
        file << "X T Z" << endl;
        if(file) {
            for (int t_idx = 0; t_idx < t.size(); t_idx++) {
                for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                    file << x(x_idx) << " " << t(t_idx) << " " << src(t_idx, x_idx) << endl;
                }
            }
            file.close();
        }
    }

    void analytic_solution() {
        U_analytic = MatrixXcd(N + 1, M + 1);
//        for (int t_idx = 0; t_idx < t.size(); t_idx++) {
//            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
//                U_analytic(t_idx, x_idx) = A * exp(complex<double>(0.0, 1.0) * (B * x(x_idx) - (A * A * lambda + B * B) * t(t_idx) + C));
//            }
//        }
        // B-soliton
        for (int t_idx = 0; t_idx < t.size(); t_idx++) {
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                complex<double> _tmp1 = (1.0 + 2.0 * complex<double>(0.0, 1.0) * t(t_idx)) * cosh(x(x_idx)) - x(x_idx) * sinh(x(x_idx));
                complex<double> _tmp2 = 1.0 + 2.0 * x(x_idx) * x(x_idx) + 8.0 * t(t_idx) * t(t_idx) + cosh(2.0 * x(x_idx));
                complex<double> _tmp3 = 4.0 * exp(complex<double>(0.0, 1.0) * t(t_idx));
                U_analytic(t_idx, x_idx) = _tmp3 * _tmp1 / _tmp2;
            }
        }
    }

    VectorXd absolute_error() {
        VectorXd error_vector = VectorXd(U.rows());
        MatrixXcd error_matrix = U_analytic - U;
        for (int t_idx = 0; t_idx < t.size(); t_idx++) {
            double absolute_error = 0.0;
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                absolute_error += abs(error_matrix(t_idx, x_idx).real());
                absolute_error += abs(error_matrix(t_idx, x_idx).imag());
            }
            error_vector(t_idx) = absolute_error / (2.0 * (double)x.size());
        }
        return error_vector;
    }

    void solve() {
        compute();
        write_real_matrix_to_file(U, "../besse/real.txt");
        write_imag_matrix_to_file(U, "../besse/imag.txt");
        write_abs_matrix_to_file(U, "../besse/abs.txt");
        analytic_solution();
        write_real_matrix_to_file(U_analytic, "../besse/analytic_real.txt");
        write_imag_matrix_to_file(U_analytic, "../besse/analytic_imag.txt");
        write_abs_matrix_to_file(U_analytic, "../besse/analytic_abs.txt");
    }

    void compute() {
        VectorXcd V_minus = VectorXcd(M + 1);
        VectorXcd V_plus = VectorXcd(M + 1);

//        V_minus = U.row(0).cwiseAbs2(); // V^-1/2 = |U|^2
        for (int x_idx = 0; x_idx < U.cols(); x_idx++) {
            V_minus(x_idx) = abs(U(0, x_idx)) * abs(U(0, x_idx));
        }

        double r = k / (2.0 * h * h);
        cout << "r: " << r << endl;


        for (int n = 0; n < N; n++) { // 0 -> N-1 / Считаем U от 1 до N
//            cout << "BESSE: STEP " << n << "/" << N-1 << endl;
            // Вычисление V_plus
            VectorXcd _tmp_vector1 = -V_minus;
            VectorXcd _tmp_vector3 = VectorXcd(U.cols());
            for (int x_idx = 0; x_idx < U.cols(); x_idx++) {
                _tmp_vector3(x_idx) = 2.0 * abs(U(n, x_idx)) * abs(U(n, x_idx));
            }
            //VectorXcd _tmp_vector2 = 2.0 * U.row(n).cwiseAbs2();
            VectorXcd _tmp_vector2;
            V_plus = _tmp_vector1 + _tmp_vector3;

            MatrixXcd A_plus = MatrixXcd(M + 1, M + 1);
            // Первая строка
            A_plus(0, 0) = complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(0);
            A_plus(0, 1) = 1.0;
            A_plus(0, M) = 1.0;
            // Последняя строка
            A_plus(M, 0) = 1.0;
            A_plus(M, M - 1) = 1.0;
            A_plus(M, M) = complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(M);
            // Промежуточные строки
            for (int m = 1; m < M; m++) {
                complex<double> a_plus = complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(m);
                A_plus(m, m - 1) = 1.0;
                A_plus(m, m) = a_plus;
                A_plus(m, m + 1) = 1.0;
            }
            A_plus = r * A_plus;

            MatrixXcd A_minus = MatrixXcd(M + 1, M + 1);
            // Первая строка
            A_minus(0, 0) = -complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(0);
            A_minus(0, 1) = 1.0;
            A_minus(0, M) = 1.0;
            // Последняя строка
            A_minus(M, 0) = 1.0;
            A_minus(M, M - 1) = 1.0;
            A_minus(M, M) = -complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(M);
            // Промежуточные строки
            for (int m = 1; m < M; m++) {
                complex<double> a_minus = -complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(m);
                A_minus(m, m - 1) = 1.0;
                A_minus(m, m) = a_minus;
                A_minus(m, m + 1) = 1.0;
            }
            A_minus = -r * A_minus;

            _tmp_vector1 = U.row(n);
            MatrixXcd _tmp_matrix1 = A_minus * _tmp_vector1;
            MatrixXcd _tmp_matrix2 = A_plus.inverse();
            _tmp_vector2 = _tmp_matrix2 * _tmp_matrix1;
            U.row(n + 1) = _tmp_vector2;

            // Следующий шаг
            V_minus = V_plus;
        }
    }
};


int main()
{
    Besse* besse = new Besse(100,
                        1000,
                        0.0,
                        1.0,
                        -2.0,
                        [](double x_val) -> complex<double> {return exp(complex<double>(0.0, 1.0) * x_val);}, // e^(ix) - начальная функция
                        1.0,
                        1.0,
                        0.0);
    besse->solve();
    cout << besse->absolute_error() << endl;

}