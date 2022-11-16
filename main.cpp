#include <iostream>
#include <chrono>
#include <vector>
#include <complex>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/LU>
#include <fstream>
#include <memory>

using namespace std;
using namespace Eigen;

class Besse {
public:
    MatrixXcd U; // Решение в точке [t_n][x_m]
    MatrixXcd U_analytic;
    MatrixXd error_matrix;

    VectorXd x; // Сетка x
    double x_start = -M_PI; // x начала
    double x_stop = M_PI; // x конца
    int M; // Количество разбиений по x
    double h = 0.0; // Шаг по x

    VectorXd t; // Сетка t
    double t_start = 0.0; // t начала
    double t_stop = 1.0; // t конца
    int N; // Количество разбиений по t
    double k = 0.0; // Шаг по t

    double lambda;
    double A, B, C; // Коэффициенты аналитического решения для f(x) = exp(ix)
    bool is_bi_soliton = false;
    bool is_exp_ix = false;
    bool is_sin_x = false;

    bool has_analytic = false;

    void init() {
        prepare_x();
        prepare_t();
        prepare_u();
    }

    void prepare_x() {
        cout << "BESSE: prepare X" << endl;
        x = VectorXd::Zero(M + 1);
        h = (x_stop - x_start) / (M + 1);
        cout << "h: " << h << endl;
        for (int m = 0; m <= M; m++) {
            x[m] = x_start + m * h;
        }
    }

    void prepare_t() {
        cout << "BESSE: prepare T" << endl;
        t = VectorXd::Zero(N + 1);
        k = (t_stop - t_start) / (N + 1);
        cout << "k: " << k << endl;
        for (int n = 0; n <= N; n++) {
            t[n] = t_start + n * k;
        }
    }

    void prepare_u() {
        if (is_bi_soliton) {
            prepare_u_bi_soluton();
        }
        if (is_exp_ix) {
            prepare_u_exp_ix();
        }
        if (is_sin_x) {
            prepare_u_sin_x();
        }
    }

    void prepare_u_bi_soluton() {
        cout << "BESSE: prepare U Bi-Soliton" << endl;
        U = MatrixXcd::Zero(N + 1, M + 1);
        for (int x_idx = 0; x_idx < x.size(); x_idx++) {
            complex<double> _tmp1 = (1.0 + 2.0 * complex<double>(0.0, 1.0) * t(0)) * cosh(x(x_idx)) - x(x_idx) * sinh(x(x_idx));
            complex<double> _tmp2 = 1.0 + 2.0 * x(x_idx) * x(x_idx) + 8.0 * t(0) * t(0) + cosh(2.0 * x(x_idx));
            complex<double> _tmp3 = 4.0 * exp(complex<double>(0.0, 1.0) * t(0));
            complex<double> _tmp4 = _tmp3 * _tmp1 / _tmp2;
            U(0, x_idx) = _tmp4;
        }
    }

    void prepare_u_exp_ix() {
        cout << "BESSE: prepare U exp(ix)" << endl;
        U = MatrixXcd::Zero(N + 1, M + 1);
        for (int x_idx = 0; x_idx < x.size(); x_idx++) {
            U(0, x_idx) = exp(complex<double>(0.0, 1.0) * complex<double>(x(x_idx), 0.0));
        }
    }

    void prepare_u_sin_x() {
        cout << "BESSE: prepare U sin(x)" << endl;
        U = MatrixXcd::Zero(N + 1, M + 1);
        for (int x_idx = 0; x_idx < x.size(); x_idx++) {
            U(0, x_idx) = complex<double>(sin(x(x_idx)), 0.0);
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
        MatrixXd abs_matrix = MatrixXd::Zero(src.rows(), src.cols());
        for (int t_idx = 0; t_idx < t.size(); t_idx++) {
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
               abs_matrix(t_idx, x_idx) = abs(src(t_idx, x_idx));
            }
        }
        write_matrix_to_file(abs_matrix, path_and_name);
    }

    void write_matrix_to_file(const MatrixXd& src, const string& path_and_name) {
        cout << "BESSE: write matrix " << path_and_name << endl;
        ofstream file(path_and_name, ios::out | ios::trunc);
        if(file) {
            file << "X T Z" << endl;
            for (int t_idx = 0; t_idx < t.size(); t_idx++) {
                for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                    file << x(x_idx) << " " << t(t_idx) << " " << src(t_idx, x_idx) << endl;
                }
            }
            file.close();
        }
    }

    void analytic_solution() {
        if (is_bi_soliton) {
            analytic_solution_bi_soliton();
            has_analytic = true;
        }
        if (is_exp_ix) {
            analytic_solution_exp_ix();
            has_analytic = true;
        }
    }

    void analytic_solution_exp_ix() {
        cout << "BESSE: analytic solution exp(ix)" << endl;
        U_analytic = MatrixXcd::Zero(N + 1, M + 1);
        for (int t_idx = 0; t_idx < t.size(); t_idx++) {
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                U_analytic(t_idx, x_idx) = A * exp(complex<double>(0.0, 1.0) * (B * x(x_idx) - (A * A * lambda + B * B) * t(t_idx) + C));
            }
        }
    }

    void analytic_solution_bi_soliton() {
        cout << "BESSE: analytic solution bi-soliton" << endl;
        U_analytic = MatrixXcd::Zero(N + 1, M + 1);
        for (int t_idx = 0; t_idx < t.size(); t_idx++) {
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                complex<double> _tmp1 = (1.0 + 2.0 * complex<double>(0.0, 1.0) * t(t_idx)) * cosh(x(x_idx)) - x(x_idx) * sinh(x(x_idx));
                complex<double> _tmp2 = 1.0 + 2.0 * x(x_idx) * x(x_idx) + 8.0 * t(t_idx) * t(t_idx) + cosh(2.0 * x(x_idx));
                complex<double> _tmp3 = 4.0 * exp(complex<double>(0.0, 1.0) * t(t_idx));
                U_analytic(t_idx, x_idx) = _tmp3 * _tmp1 / _tmp2;
            }
        }
    }

    void compute_absolute_error() {
        cout << "BESSE: compute absolute error" << endl;
        error_matrix = MatrixXd::Zero(N + 1, M + 1);
        for (int t_idx = 0; t_idx < t.size(); t_idx++) {
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                error_matrix(t_idx, x_idx) = abs(U_analytic(t_idx, x_idx) - U(t_idx, x_idx));
            }
        }
    }

    void solve(const string& folder) {
        cout << "BESSE: in solve" << endl;
        init();
        compute();
        write_real_matrix_to_file(U, folder + "/real.txt");
        write_imag_matrix_to_file(U, folder + "/imag.txt");
        write_abs_matrix_to_file(U, folder + "/abs.txt");
        analytic_solution();
        if (has_analytic) {
            write_real_matrix_to_file(U_analytic, folder + "/real_analytic.txt");
            write_imag_matrix_to_file(U_analytic, folder + "/imag_analytic.txt");
            write_abs_matrix_to_file(U_analytic, folder + "/abs_analytic.txt");
            compute_absolute_error();
            write_matrix_to_file(error_matrix, folder + "/error_matrix.txt");
        }
    }

    void compute() {
        cout << "BESSE: in compute" << endl;
        VectorXcd V_minus = VectorXcd::Zero(M + 1);
        VectorXcd V_plus = VectorXcd::Zero(M + 1);

        // V^-1/2 = |U|^2
        for (int x_idx = 0; x_idx < x.size(); x_idx++) {
            V_minus(x_idx) = abs(U(0, x_idx)) * abs(U(0, x_idx));
        }

        complex<double> r = complex<double>(k / (2.0 * h * h), 0.0);
        cout << "r: " << r << endl;

        for (int n = 0; n < N; n++) { // 0 -> N-1 / Считаем U от 1 до N
            if (n % 10 == 0) {
                cout << "BESSE: STEP " << n << "/" << N - 1 << endl;
            }

            // Вычисление V_plus
            VectorXcd _tmp_vector1 = -V_minus;
            VectorXcd _tmp_vector2 = VectorXcd::Zero(x.size());
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                _tmp_vector2(x_idx) = 2.0 * abs(U(n, x_idx)) * abs(U(n, x_idx));
            }
            V_plus = _tmp_vector1 + _tmp_vector2;

            MatrixXcd A_plus = MatrixXcd::Zero(M + 1, M + 1);
            // Первая строка
            A_plus(0, 0) = complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(0);
            A_plus(0, 1) =  complex<double>(1.0, 0.0);
            A_plus(0, M) = complex<double>(1.0, 0.0);
            // Последняя строка
            A_plus(M, 0) = complex<double>(1.0, 0.0);
            A_plus(M, M - 1) = complex<double>(1.0, 0.0);
            A_plus(M, M) = complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(M);
            // Промежуточные строки
            for (int m = 1; m < M; m++) {
                complex<double> a_plus = complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(m);
                A_plus(m, m - 1) = complex<double>(1.0, 0.0);
                A_plus(m, m) = a_plus;
                A_plus(m, m + 1) = complex<double>(1.0, 0.0);
            }
            A_plus = r * A_plus;

            MatrixXcd A_minus = MatrixXcd::Zero(M + 1, M + 1);
            // Первая строка
            A_minus(0, 0) = -complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(0);
            A_minus(0, 1) = complex<double>(1.0, 0.0);
            A_minus(0, M) = complex<double>(1.0, 0.0);
            // Последняя строка
            A_minus(M, 0) = complex<double>(1.0, 0.0);
            A_minus(M, M - 1) = complex<double>(1.0, 0.0);
            A_minus(M, M) = -complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(M);
            // Промежуточные строки
            for (int m = 1; m < M; m++) {
                complex<double> a_minus = -complex<double>(0.0, 1.0) / r - 2.0 - lambda * h * h * V_plus(m);
                A_minus(m, m - 1) = complex<double>(1.0, 0.0);
                A_minus(m, m) = a_minus;
                A_minus(m, m + 1) = complex<double>(1.0, 0.0);
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

class BesseHelper {
public:
    static void compute_bi_soliton_500_500() {
        unique_ptr<Besse> besse(new Besse);
        besse->is_bi_soliton = true;
        besse->M = 500;
        besse->N = 500;
        besse->t_start = -5.0;
        besse->t_stop = 5.0;
        besse->x_start = -3.0 * M_PI;
        besse->x_stop = 3.0 * M_PI;
        besse->lambda = -2.0;

        besse->init();
        besse->solve("../data/bi_soliton_500_500");
    }

    static void compute_bi_soliton_1000_1000() {
        unique_ptr<Besse> besse(new Besse);
        besse->is_bi_soliton = true;
        besse->M = 1000;
        besse->N = 1000;
        besse->t_start = -5.0;
        besse->t_stop = 5.0;
        besse->x_start = -3.0 * M_PI;
        besse->x_stop = 3.0 * M_PI;
        besse->lambda = -2.0;

        besse->init();
        besse->solve("../data/bi_soliton_1000_1000");
    }

    static void compute_exp_ix_400_400() {
        unique_ptr<Besse> besse(new Besse);
        besse->is_exp_ix = true;
        besse->M = 400;
        besse->N = 400;
        besse->t_start = 0.0;
        besse->t_stop = 5.0;
        besse->x_start = -M_PI;
        besse->x_stop = M_PI;
        besse->lambda = 1.0;
        besse->A = 1.0;
        besse->B = 1.0;
        besse->C = 0.0;

        besse->init();
        besse->solve("../data/exp_ix_400_400");
    }

    static void compute_sin_x_500_1000() {
        unique_ptr<Besse> besse(new Besse);
        besse->is_sin_x = true;
        besse->M = 500;
        besse->N = 1000;
        besse->t_start = 0.0;
        besse->t_stop = 5.0;
        besse->x_start = -M_PI;
        besse->x_stop = M_PI;
        besse->lambda = -2.0;

        besse->init();
        besse->solve("../data/sin_x_500_1000");
    }
};


int main(){
    auto begin = chrono::steady_clock::now();

    BesseHelper::compute_bi_soliton_1000_1000();

    auto end = chrono::steady_clock::now();
    auto elapsed_m = std::chrono::duration_cast<chrono::minutes>(end - begin);
    cout << "Время работы: " << elapsed_m.count() << " минут" << endl;
}