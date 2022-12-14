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

    // Коэффициенты новой задачи
    double m = 7.0;
    double gamma0 = 4.32e-12;
    double chi = 20.0;
    double E0 = 0.0;
    double omega = 5.0e14;
    double omega0 = 1.0e14;
    double Kb = 1.38e-16;
    double T = 77.0;
    double t_coef = 0.0; // Не понятно, что это
    double a = 0.3;
    double b = 2.0;

    // Внутренние переменные
    const int num_splits = 20;

    static double factorial(const int n) {
        double f = 1;
        for (int i=1; i<=n; ++i) f *= i;
        return f;
    }

    static double simpsonIntegral(double a, double b, int n, const std::function<double (double)> &f) {
        const double width = (b-a)/n;
        double simpson_integral = 0;
        for(int step = 0; step < n; step++) {
            const double x1 = a + step*width;
            const double x2 = a + (step+1)*width;
            simpson_integral += (x2-x1)/6.0*(f(x1) + 4.0*f(0.5*(x1+x2)) + f(x2));
        }
        return simpson_integral;
    }

    double eps(double p, double s) {
        return gamma0 * sqrt(1.0 + 4.0 * cos(p) * cos(M_PI * s / m) + 4.0 * cos(M_PI * s / m) * cos(M_PI * s / m));
    }

    double delta(double alpha, double s) {
        return simpsonIntegral(-M_PI, M_PI, num_splits, [alpha, this](double p) {return eps(p, alpha) * cos(p * alpha);}) / M_PI;
    }

    double G(double alpha, double s) {
        double nominator = simpsonIntegral(-M_PI, M_PI, num_splits, [this, &s, &alpha](double r) {
            double sum = 0.0;
            for (int i = 1; i <= 9; i++) {
                sum += delta(i, s) * cos(i * r) / (Kb * T);
            }
            return cos(alpha * r) * exp(-sum);
        });
        double denominator = simpsonIntegral(-M_PI, M_PI, num_splits, [this, &s, &alpha](double r) {
            double sum = 0.0;
            for (int i = 1; i <= 9; i++) {
                sum += delta(i, s) * cos(i * r) / (Kb * T);
            }
            return exp(-sum);
        });
        return -alpha * delta(alpha, s) * nominator / (gamma0 * denominator);
    }

    double G1(double alpha, double s) {
        return G(alpha, s) * cos(alpha * E0 * t_coef);
    }

    double G2(double alpha, double s) {
        return G(alpha, s) * sin(alpha * E0 * t_coef);
    }

    void prepare_x() {
        cout << "BESSE: prepare X" << endl;
        x = VectorXd::Zero(M + 1);
        h = (x_stop - x_start) / (M + 1);
        cout << "h: " << h << endl;
        for (int m_idx = 0; m_idx <= M; m_idx++) {
            x[m_idx] = x_start + m_idx * h;
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
        cout << "BESSE: prepare U" << endl;
        U = MatrixXcd::Zero(N + 1, M + 1);
        for (int x_idx = 0; x_idx < x.size(); x_idx++) {
            double v0 = (x_start + x_stop) / 2.0;
            U(0, x_idx) = a * exp(-(x(x_idx) - v0) * (x(x_idx) - v0) / (b * b));
        }
    }

    void init() {
        prepare_x();
        prepare_t();
        prepare_u();
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

    void write_square_abs_matrix_to_file(const MatrixXcd& src, const string& path_and_name) {
        MatrixXd abs_matrix = MatrixXd::Zero(src.rows(), src.cols());
        for (int t_idx = 0; t_idx < t.size(); t_idx++) {
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                abs_matrix(t_idx, x_idx) = abs(src(t_idx, x_idx)) * abs(src(t_idx, x_idx));
            }
        }
        write_matrix_to_file(abs_matrix, path_and_name);
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

    void compute() {
        cout << "BESSE: in compute" << endl;

        vector<VectorXcd> phi_abs_minus = {
                VectorXcd::Zero(M + 1), // |Ф|^0
                VectorXcd::Zero(0), // |Ф|^1
                VectorXcd::Zero(M + 1), // |Ф|^2
                VectorXcd::Zero(0), // |Ф|^3
                VectorXcd::Zero(M + 1), // |Ф|^4
                VectorXcd::Zero(0), // |Ф|^5
                VectorXcd::Zero(M + 1), // |Ф|^6
                VectorXcd::Zero(0), // |Ф|^7
                VectorXcd::Zero(M + 1), // |Ф|^8
                VectorXcd::Zero(0), // |Ф|^9
                VectorXcd::Zero(M + 1), // |Ф|^10
                VectorXcd::Zero(0), // |Ф|^11
                VectorXcd::Zero(M + 1) // |Ф|^12
        };
        vector<VectorXcd> phi_abs_plus = {
                VectorXcd::Zero(M + 1), // |Ф|^0
                VectorXcd::Zero(0), // |Ф|^1
                VectorXcd::Zero(M + 1), // |Ф|^2
                VectorXcd::Zero(0), // |Ф|^3
                VectorXcd::Zero(M + 1), // |Ф|^4
                VectorXcd::Zero(0), // |Ф|^5
                VectorXcd::Zero(M + 1), // |Ф|^6
                VectorXcd::Zero(0), // |Ф|^7
                VectorXcd::Zero(M + 1), // |Ф|^8
                VectorXcd::Zero(0), // |Ф|^9
                VectorXcd::Zero(M + 1), // |Ф|^10
                VectorXcd::Zero(0), // |Ф|^11
                VectorXcd::Zero(M + 1) // |Ф|^12
        };

        // V^-1/2 = |U|^power
        for (int power = 0; power <= 12; power++) {
            if (power % 2 == 0) {
                for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                    phi_abs_minus[power](x_idx) = pow(abs(U(0, x_idx)), power);
                }
            }
        }

        complex<double> r = complex<double>(k / (2.0 * h * h), 0.0);
        cout << "r: " << r << endl;

        for (int n = 0; n < N; n++) { // 0 -> N-1 / Считаем U от 1 до N
            //if (n % 10 == 0) {
                cout << "BESSE: STEP " << n << "/" << N - 1 << endl;
            //}

            //cout << n << ": " << U.row(n) << endl;

            // Вычисление V_plus
            for (int power = 0; power <= 12; power++) {
                if (power % 2 == 0) {
                    VectorXcd _tmp_vector1 = -phi_abs_minus[power];
                    VectorXcd _tmp_vector2 = VectorXcd::Zero(x.size());
                    for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                        _tmp_vector2(x_idx) = 2.0 * pow(abs(U(n, x_idx)), power);
                    }
                    phi_abs_plus[power] = _tmp_vector1 + _tmp_vector2;
                    //cout << power << " : "<< phi_abs_plus[power] << endl;
                }
            }


            VectorXcd C = VectorXcd::Zero(M + 1);
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                complex<double> outer_sum = 0.0;
                for (int alpha = 1; alpha <= 9; alpha++) {
                    complex<double> middle_sum = 0.0;
                    for (int s = 1; s <= 7; s++) {
                        complex<double> inner_sum = 0.0;
                        for (int l = 0; l <= 5; l++) {
                            inner_sum += pow(-1.0, l) * pow(alpha, 2 * l) * phi_abs_plus[2 * l](x_idx) / (factorial(l) * factorial(l + 1) * pow(2, 2 * l));
                        }
                        middle_sum += G1(alpha, s) * inner_sum;
                    }
                    outer_sum += (double)alpha * middle_sum;
                }
                C(x_idx) = outer_sum;
            }
            //cout << n << ": " << C << endl;


            VectorXcd B = VectorXcd::Zero(M + 1);
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                complex<double> outer_sum = 0.0;
                for (int alpha = 1; alpha <= 9; alpha++) {
                    complex<double> middle_sum = 0.0;
                    for (int s = 1; s <= 7; s++) {
                        complex<double> inner_sum_1 = 0.0;
                        for (int l = 0; l <= 5; l++) {
                            inner_sum_1 += pow(-1.0, l) * pow(alpha, 2 * l) * phi_abs_plus[2 * l](x_idx) / (factorial(l) * factorial(l) * pow(2, 2 * l));
                        }
                        complex<double> inner_sum_2 = 0.0;
                        for (int l = 0; l <= 5; l++) {
                            inner_sum_2 += pow(-1.0, l) * pow(alpha, 2 * l + 2) * phi_abs_plus[2 * l + 2](x_idx) / (factorial(l) * factorial(l + 2) * pow(2, 2 * l + 2));
                        }
                        middle_sum += G2(alpha, s) * (inner_sum_1 + inner_sum_2);
                    }
                    outer_sum += middle_sum;
                }
                B(x_idx) = outer_sum;
            }

            MatrixXcd A_plus = MatrixXcd::Zero(M + 1, M + 1);
            // Первая строка
            A_plus(0, 0) = 2.0 * chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(0);
            A_plus(0, 1) =  complex<double>(1.0, 0.0);
            A_plus(0, M) = complex<double>(1.0, 0.0);
            // Последняя строка
            A_plus(M, 0) = complex<double>(1.0, 0.0);
            A_plus(M, M - 1) = complex<double>(1.0, 0.0);
            A_plus(M, M) = 2.0 * chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(M);
            // Промежуточные строки
            for (int m_idx = 1; m_idx < M; m_idx++) {
                A_plus(m_idx, m_idx - 1) = complex<double>(1.0, 0.0);
                A_plus(m_idx, m_idx) = 2.0 * chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(m_idx);
                A_plus(m_idx, m_idx + 1) = complex<double>(1.0, 0.0);
            }
            A_plus = r * A_plus;

            MatrixXcd A_minus = MatrixXcd::Zero(M + 1, M + 1);
            // Первая строка
            A_minus(0, 0) = -2.0 * chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(0);
            A_minus(0, 1) = complex<double>(1.0, 0.0);
            A_minus(0, M) = complex<double>(1.0, 0.0);
            // Последняя строка
            A_minus(M, 0) = complex<double>(1.0, 0.0);
            A_minus(M, M - 1) = complex<double>(1.0, 0.0);
            A_minus(M, M) = -2.0 * chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(M);
            // Промежуточные строки
            for (int m_idx = 1; m_idx < M; m_idx++) {
                A_minus(m_idx, m_idx - 1) = complex<double>(1.0, 0.0);
                A_minus(m_idx, m_idx) = -2.0 * chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(m_idx);
                A_minus(m_idx, m_idx + 1) = complex<double>(1.0, 0.0);
            }
            A_minus = -r * A_minus;

//            MatrixXcd B_matrix = MatrixXcd::Zero(M + 1, M + 1);
//            for (int m = 0; m <= M; m++) {
//                B_matrix(m, m) = B(m);
//            }

            VectorXcd _tmp_vector1 = U.row(n);
            MatrixXcd _tmp_matrix1 = A_minus * _tmp_vector1;
            //_tmp_matrix1 = _tmp_matrix1 + B_matrix;
            MatrixXcd _tmp_matrix2 = A_plus.inverse();
            VectorXcd _tmp_vector2 = _tmp_matrix2 * _tmp_matrix1;
            VectorXcd _tmp_vector3 = _tmp_matrix2 * B;
            U.row(n + 1) = _tmp_vector2 + _tmp_vector3;

            // Следующий шаг
            phi_abs_minus = phi_abs_plus;
        }
    }

    void solve(const string& folder) {
        cout << "BESSE: in solve" << endl;
        init();
        compute();
        write_real_matrix_to_file(U, folder + "/real.txt");
        write_imag_matrix_to_file(U, folder + "/imag.txt");
        write_abs_matrix_to_file(U, folder + "/abs.txt");
        write_square_abs_matrix_to_file(U, folder + "/abs_square.txt");
    }
};

class BesseHelper {
public:
    static void compute_nush_analogue() {
        unique_ptr<Besse> besse(new Besse);
        besse->M = 100;
        besse->N = 100;
        besse->t_start = 0.0;
        besse->t_stop = 5.0;
        besse->x_start = -M_PI;
        besse->x_stop = M_PI;
        besse->solve("../data/nush_analogue");
    }
};


int main(){
    Eigen::setNbThreads(6);
    auto begin = chrono::steady_clock::now();

    BesseHelper::compute_nush_analogue();

    auto end = chrono::steady_clock::now();
    auto elapsed_m = std::chrono::duration_cast<chrono::minutes>(end - begin);
    cout << "Время работы: " << elapsed_m.count() << " минут" << endl;
}