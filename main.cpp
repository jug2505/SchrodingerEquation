#include <iostream>
#include <chrono>
#include <vector>
#include <complex>
#include <cmath>
#include <map>

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

    // Кэш
    map<pair<int, int>, double> G_alpha_s_cache;
    map<pair<int, int>, double> delta_alpha_s_cache;

    // Коэффициенты новой задачи
    double m = 7.0;
    double gamma0 = 4.32e-12;
    double chi = 20.0;
    double E0 = 0.0;
    double omega = 5.0e14;
    double omega0 = 1.0e14;
    double Kb = 1.38e-16;
    double T = 77.0;
    double a = 0.3;
    double b = 2.0;

    // Внутренние переменные
    const int num_splits = 100000;
    int write_t_splits = 4;
    int current_t_split = 0;

    static double factorial(const int n) {
        double f = 1;
        for (int i=1; i<=n; ++i) f *= i;
        return f;
    }

    static double simpsonIntegral(const double a, const double b, const int n, const std::function<double (double)> &f) {
        const double width = (b-a)/n;
        double simpson_integral = 0;
        for(int step = 0; step < n; step++) {
            const double x1 = a + step*width;
            const double x2 = a + (step+1)*width;
            simpson_integral += (x2-x1)/6.0*(f(x1) + 4.0*f(0.5*(x1+x2)) + f(x2));
        }
        return simpson_integral;
    }

    double eps(const double p, const double s) {
        return gamma0 * sqrt(1.0 + 4.0 * cos(p) * cos(M_PI * s / m) + 4.0 * cos(M_PI * s / m) * cos(M_PI * s / m));
    }

    double delta(const int alpha, const int s) {
        if (delta_alpha_s_cache.find({alpha, s}) != delta_alpha_s_cache.end()) {
            return delta_alpha_s_cache[{alpha, s}];
        }
        double result = simpsonIntegral(-M_PI, M_PI, num_splits, [alpha, s, this](double p) {return eps(p, s) * cos(p * alpha);}) / M_PI;
        delta_alpha_s_cache[{alpha, s}] = result;
        cout << "BESSE: delta_alpha_s alpha = " << alpha << ", s = " << s << " cached" << endl;
        return result;
    }

    double G(const int alpha, const int s) {
        if (G_alpha_s_cache.find({alpha, s}) != G_alpha_s_cache.end()) {
            return G_alpha_s_cache[{alpha, s}];
        }

        double nominator = simpsonIntegral(-M_PI, M_PI, num_splits, [this, &s, &alpha](double r) {
            double sum = 0.0;
            for (int i = 1; i <= 9; i++) {
                sum += delta(i, s) * cos(i * r) / (Kb * T);
            }
            return cos(alpha * r) * exp(-sum);
        });

        double denominator = 0.0;
        for (int s_idx = 1; s_idx <= 7; s_idx++) {
            denominator += simpsonIntegral(-M_PI, M_PI, num_splits, [this, &s_idx, &alpha](double r) {
                double sum = 0.0;
                for (int i = 1; i <= 9; i++) {
                    sum += delta(i, s_idx) * cos(i * r) / (Kb * T);
                }
                return exp(-sum);
            });
        }
        double result = -alpha * delta(alpha, s) * nominator / (gamma0 * denominator);
        G_alpha_s_cache[{alpha, s}] = result;
        cout << "BESSE: G_alpha_s alpha = " << alpha << ", s = " << s << " cached" << endl;
        
        return result;
    }

    double G1(const int alpha, const int s, const double t_value) {
        return G(alpha, s) * cos(alpha * E0 * t_value * 0.0001);
    }

    double G2(const int alpha, const int s, const double t_value) {
        return G(alpha, s) * sin(alpha * E0 * t_value * 0.0001);
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
                abs_matrix(t_idx, x_idx) = abs(src(t_idx, x_idx)) * abs(src(t_idx, x_idx)) / (a * a);
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

    void write_cache_to_file(const map<pair<int, int>, double>& map_cache, const string& path_and_name) {
        cout << "BESSE: write cache matrix " << path_and_name << endl;
        ofstream file(path_and_name, ios::out | ios::trunc);
        if(file) {
            file << "alpha s value" << endl;
            for (auto const& [key, val] : map_cache) {
                file << key.first << " " << key.second << " " << val << endl;
            }
            file.close();
        }
    }

    void write_G_cache_sum_to_file(const string& path_and_name) {
        cout << "BESSE: write G cache sum to " << path_and_name << endl;
        ofstream file(path_and_name, ios::out | ios::trunc);
        if(file) {
            file << "alpha s_sum" << endl;
            for (int alpha_idx = 1; alpha_idx <= 9; alpha_idx++){
                double sum = 0.0;
                for (int s_idx = 1; s_idx <= 7; s_idx++){
                    sum += G_alpha_s_cache[{alpha_idx, s_idx}];
                }
                file << alpha_idx << " " << sum << endl;
            }
            file.close();
        }
    }

    void write_t_splits_to_file(const string& folder, int n) {
        if (n != current_t_split * N / write_t_splits && n != N-1) return;
        if (n == N-1) n = N;

        ofstream file(folder + "/T=" + to_string(t(n)) + ".txt", ios::out | ios::trunc);
        if(file) {
            file << "X Z" << endl;
            for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                file << x(x_idx) << " " << abs(U(n, x_idx)) * abs(U(n, x_idx)) / (a * a) << endl;
            }
            file.close();
        }

        current_t_split++;
    }

    void write_beam_width_to_file(const string& path_and_name) {
        ofstream file(path_and_name, ios::out | ios::trunc);
        if(file) {
            file << "z d" << endl;
            for (int t_idx = 0; t_idx < t.size(); t_idx++) {
                long x_idx = x.size() % 2 == 0 ? x.size() / 2 : x.size() / 2 + 1;
                double middle_value = abs(U(t_idx,x_idx)) * abs(U(t_idx,x_idx)) / (a * a);
                x_idx++;
                double next_value = abs(U(t_idx,x_idx)) * abs(U(t_idx,x_idx)) / (a * a);
                while (next_value * 2.0 > middle_value) {
                    x_idx++;
                    next_value = abs(U(t_idx,x_idx)) * abs(U(t_idx,x_idx)) / (a * a);
                }
                file << t(t_idx) << " " << x(x_idx) << endl;
            }
            file.close();
        }
    }

    void compute(const string& folder) {
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
            if (n % 10 == 0) {
                cout << "BESSE: STEP " << n << "/" << N - 1 << endl;
            }

            // Вычисление V_plus
            for (int power = 0; power <= 12; power++) {
                if (power % 2 == 0) {
                    VectorXcd _tmp_vector1 = -phi_abs_minus[power];
                    VectorXcd _tmp_vector2 = VectorXcd::Zero(x.size());
                    for (int x_idx = 0; x_idx < x.size(); x_idx++) {
                        _tmp_vector2(x_idx) = 2.0 * pow(abs(U(n, x_idx)), power);
                    }
                    phi_abs_plus[power] = _tmp_vector1 + _tmp_vector2;
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
                        middle_sum += G1(alpha, s, t(n + 1)) * inner_sum;
                    }
                    outer_sum += (double)alpha * middle_sum;
                }
                C(x_idx) = outer_sum;
            }

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
                        middle_sum += G2(alpha, s, t(n + 1)) * (inner_sum_1 + inner_sum_2);
                    }
                    outer_sum += middle_sum;
                }
                B(x_idx) = outer_sum;
            }

            MatrixXcd A_plus = MatrixXcd::Zero(M + 1, M + 1);
            // Первая строка
            A_plus(0, 0) = chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(0);
            A_plus(0, 1) =  complex<double>(1.0, 0.0);
            A_plus(0, M) = complex<double>(1.0, 0.0);
            // Последняя строка
            A_plus(M, 0) = complex<double>(1.0, 0.0);
            A_plus(M, M - 1) = complex<double>(1.0, 0.0);
            A_plus(M, M) = chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(M);
            // Промежуточные строки
            for (int m_idx = 1; m_idx < M; m_idx++) {
                A_plus(m_idx, m_idx - 1) = complex<double>(1.0, 0.0);
                A_plus(m_idx, m_idx) = chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(m_idx);
                A_plus(m_idx, m_idx + 1) = complex<double>(1.0, 0.0);
            }
            A_plus = r * A_plus;

            MatrixXcd A_minus = MatrixXcd::Zero(M + 1, M + 1);
            // Первая строка
            A_minus(0, 0) = -chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(0);
            A_minus(0, 1) = complex<double>(1.0, 0.0);
            A_minus(0, M) = complex<double>(1.0, 0.0);
            // Последняя строка
            A_minus(M, 0) = complex<double>(1.0, 0.0);
            A_minus(M, M - 1) = complex<double>(1.0, 0.0);
            A_minus(M, M) = -chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(M);
            // Промежуточные строки
            for (int m_idx = 1; m_idx < M; m_idx++) {
                A_minus(m_idx, m_idx - 1) = complex<double>(1.0, 0.0);
                A_minus(m_idx, m_idx) = -chi * complex<double>(0.0, 1.0) / r - 2.0 - h * h * C(m_idx);
                A_minus(m_idx, m_idx + 1) = complex<double>(1.0, 0.0);
            }
            A_minus = -r * A_minus;

            VectorXcd _tmp_vector1 = U.row(n);
            MatrixXcd _tmp_matrix1 = A_minus * _tmp_vector1;
            //_tmp_matrix1 = _tmp_matrix1 + B_matrix;
            MatrixXcd _tmp_matrix2 = A_plus.inverse();
            VectorXcd _tmp_vector2 = _tmp_matrix2 * _tmp_matrix1;
            VectorXcd _tmp_vector3 = _tmp_matrix2 * B;
            U.row(n + 1) = _tmp_vector2 + _tmp_vector3;

            // Следующий шаг
            phi_abs_minus = phi_abs_plus;

            write_t_splits_to_file(folder, n);
        }
    }

    void solve(const string& folder) {
        cout << "BESSE: in solve" << endl;
        init();
        compute(folder);
        //write_real_matrix_to_file(U, folder + "/real.txt");
        //write_imag_matrix_to_file(U, folder + "/imag.txt");
        //write_abs_matrix_to_file(U, folder + "/abs.txt");
        write_square_abs_matrix_to_file(U, folder + "/abs_square.txt");
        write_cache_to_file(G_alpha_s_cache, folder + "/G_alpha_s.txt");
        write_cache_to_file(delta_alpha_s_cache, folder + "/delta_alpha_s.txt");
        write_G_cache_sum_to_file(folder + "/G_alpha_s_sum.txt");
        write_beam_width_to_file(folder + "/beam_width.txt");
    }
};

class BesseHelper {
public:
    static void compute_nush_analogue_1000_E0_0() {
        unique_ptr<Besse> besse(new Besse);
        besse->M = 1000;
        besse->N = 1000;
        besse->t_start = 0.0;
        besse->t_stop = 30.0;
        besse->x_start = -15.0;
        besse->x_stop = 15.0;
        besse->E0 = 0.0;
        besse->solve("../data/nush_analogue_1000_E0_0");
    }

    static void compute_nush_analogue_1000_E0_1() {
        unique_ptr<Besse> besse(new Besse);
        besse->M = 1000;
        besse->N = 1000;
        besse->t_start = 0.0;
        besse->t_stop = 30.0;
        besse->x_start = -15.0;
        besse->x_stop = 15.0;
        besse->E0 = 1.0;
        besse->solve("../data/nush_analogue_1000_E0_1");
    }

    static void compute_nush_analogue_1000_E0_01() {
        unique_ptr<Besse> besse(new Besse);
        besse->M = 1000;
        besse->N = 1000;
        besse->t_start = 0.0;
        besse->t_stop = 30.0;
        besse->x_start = -15.0;
        besse->x_stop = 15.0;
        besse->E0 = 0.1;
        besse->solve("../data/nush_analogue_1000_E0_01");
    }
};


int main(){
    Eigen::setNbThreads(6);
    auto begin = chrono::steady_clock::now();

    BesseHelper::compute_nush_analogue_1000_E0_1();
    BesseHelper::compute_nush_analogue_1000_E0_01();

    auto end = chrono::steady_clock::now();
    auto elapsed_m = std::chrono::duration_cast<chrono::minutes>(end - begin);
    cout << "Время работы: " << elapsed_m.count() << " минут" << endl;
}