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

    double eps(double p, double s) {

    }
};

class BesseHelper {
public:
    double m = 7.0;
    double gamma0 = 4.32e-12;
    double chi = 20.0;
    double E0 = 0.0;
    double omega = 5.0e14;
    double omega0 = 1.0e14;
    double Kb = 1.38e-16;
    double T = 77.0 * Kb;
    double t = 0.0; // Не понятно, что это

    int num_splits = 100;

    VectorXd phi_abs_coefficients = VectorXd::Zero(10); // Индекс - степень |Ф|

    void compute_coefficients() {

        for (int alpha = 1; alpha <= 9; alpha++) {
            for (int s = 1; s <= 7; s++) {
                for (int l = 0; l <= 5; l++) {
                    phi_abs_coefficients(2 * l) += pow(-1.0, l) * pow(alpha, 2.0 * l) / (factorial(l) * factorial(l + 1) * pow(2.0, 2.0 * l));
                    phi_abs_coefficients(2 * l) *= G1(alpha, s);
                }
            }

        }


    }

    double G1(double alpha, double s) {
        return G(alpha, s) * cos(alpha * E0 * t);
    }

    double G2(double alpha, double s) {
        return G(alpha, s) * sin(alpha * E0 * t);
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

    double delta(double alpha, double s) {
        return simpsonIntegral(-M_PI, M_PI, num_splits, [alpha, this](double p) {return eps(p, alpha) * cos(p * alpha);}) / M_PI;
    }

    double eps(double p, double s) {
        return gamma0 * sqrt(1.0 + 4.0 * cos(p) * cos(M_PI * s / m) + 4.0 * cos(M_PI * s / m) * cos(M_PI * s / m));
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

    static double factorial(const int n)
    {
        double f = 1;
        for (int i=1; i<=n; ++i)
            f *= i;
        return f;
    }

};


int main(){
    Eigen::setNbThreads(4);
    auto begin = chrono::steady_clock::now();


    auto end = chrono::steady_clock::now();
    auto elapsed_m = std::chrono::duration_cast<chrono::minutes>(end - begin);
    cout << "Время работы: " << elapsed_m.count() << " минут" << endl;
}