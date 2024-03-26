import matplotlib.pyplot as plt
import math
import time
from numpy import *
from enum import Enum


# Класс (перечисление) с типами частиц
class Type(Enum):
    DEATH = 0  # Частица, которая уже не участвует в вычислениях
    FLEX = 1  # Подвижная частица (моделируемая)
    SOLID = 2  # Твёрдая частица (граничная)


# Класс, описывающий частицу для одномерного случая
class Particle:
    def __init__(self, x, v, m, rho, energy, dv, de, type, gamma=1.4, kh=4.0/3.0):
        self.x = x
        self.v = v
        self.v_old = v
        self.m = m
        self.rho = rho
        self.energy = energy
        self.energy_old = energy
        self.dv = dv
        self.de = de
        self.type = type
        self.gamma = gamma
        self.kh = kh
        self.h = self.get_soft_length()
        self.p = self.get_pressure()
        self.c = self.get_sound_speed()

    # Расчёт давления частицы из её энергии
    def get_pressure(self):
        return self.energy * (self.gamma - 1.0) * self.rho

    # Расчёт энергии частицы из её давления
    @staticmethod
    def get_energy(p, rho, gamma):
        return p / ((gamma - 1.0) * rho)

    # Расчёт локальной скорости звука в среде
    def get_sound_speed(self):
        return sqrt(self.gamma * self.get_pressure() / self.rho)

    # Расчёт сглаживающего расстояния для частицы
    def get_soft_length(self):
        return self.kh * self.m / self.rho


# Класс - решатель для метода SPH 1D, интегрирование происходит по схеме LeapFrog (kick-drift-kick) 2-ой порядок
class SPHSolver:
    def __init__(self, particles, dt, border_dt, courant=0.6):
        self.a = 0.5
        self.b = 1.0
        self.c = 0.1
        self.dt = dt
        self.border_dt = border_dt
        self.courant = courant
        self.particles = particles

    def get_particles(self):
        return self.particles

    # Ядро SPH, в данном случае используется f - нормирующий коэффициент для одномерного случая
    def kernel(self, eps, h):
        w = 0.0
        f = 2.0 / (3.0 * h)
        eps /= h
        if eps >= 0 and eps < 1:
            w = 1.0 - 1.5 * eps * eps + 0.75 * eps * eps * eps
        elif eps >= 1 and eps < 2:
            w = 0.25 * (2.0 - eps) * (2.0 - eps) * (2.0 - eps)
        return f * w

    # Градиент ядра SPH
    def delta_kernel(self, eps, h):
        dw = 0.0
        f = 2.0 / (3.0 * h * h)
        eps = eps / h
        if eps >= 0 and eps < 1:
            dw = 2.25 * eps * eps - 3.0 * eps
        elif eps >= 1 and eps < 2:
            dw = -0.75 * (2 - eps) * (2 - eps)
        return f * dw / (eps * h)

    # Функция расчёта плотности частиц на текущей итерации, также обновление сглаживающего расстояния
    def compute_rho(self):
        rho = 0.0
        for particle in self.particles:
            if particle.type != Type.FLEX: continue
            rho = 0.0
            for neighbour in self.particles:
                if neighbour is particle: continue
                r = fabs(particle.x - neighbour.x)
                h = 0.5 * (particle.h + neighbour.h)
                if r < 2.0 * h:
                    rho += neighbour.m * self.kernel(r, h)
            particle.rho = rho + particle.m * self.kernel(0.0, particle.h)
            particle.h = particle.get_soft_length()

    # Функция расчитывающая силы попарного взаимодействия частиц
    def compute_forces(self):
        for particle in self.particles:
            if particle.type != Type.FLEX: continue
            dv = 0.0
            de = 0.0
            for neighbour in self.particles:
                if neighbour is particle: continue
                dr = particle.x - neighbour.x
                h = 0.5 * (particle.h + neighbour.h)
                r = fabs(dr)
                if r < 2.0 * h:
                    v = particle.v - neighbour.v
                    p = particle.get_pressure() / (particle.rho * particle.rho) + neighbour.get_pressure() / (neighbour.rho * neighbour.rho)
                    rh = dr * v
                    rh = h * rh / (r * r + self.c * h * h) if rh < 0.0 else 0.0
                    rho = 0.5 * (particle.rho + neighbour.rho)
                    speed = 0.5 * (particle.get_sound_speed() + neighbour.get_sound_speed())
                    viscosity = (rh * (self.b * rh - self.a * speed)) / rho if rh < 0.0 else 0.0
                    dv -= neighbour.m * (p + viscosity) * self.delta_kernel(r, h) * dr
                    de += neighbour.m * v * (p + viscosity) * self.delta_kernel(r, h) * dr
            particle.dv = dv
            particle.de = 0.5 * de

    # Шаг предиктора, получаем промежуточное состояние системы и изменяем положение частиц
    def predict(self):
        for particle in self.particles:
            if particle.type != Type.FLEX: continue
            particle.v_old = particle.v
            particle.v += self.dt * particle.dv
            particle.energy_old = particle.energy
            particle.energy += self.dt * particle.de
            particle.energy = 0.0 if particle.energy < 0.0 else particle.energy
            particle.p = particle.get_pressure()
            particle.x += self.dt * 0.5 * (particle.v_old + particle.v)

    # Шаг корректора - фиксируем состояние для следующего шага по времени
    def correct(self):
        for particle in self.particles:
            if particle.type != Type.FLEX: continue
            particle.v = 0.5 * (particle.v + particle.v_old) + particle.dv * self.dt * 0.5
            particle.energy = 0.5 * (particle.energy + particle.energy_old) + particle.de * self.dt * 0.5
            particle.energy = 0.0 if particle.energy < 0.0 else particle.energy
            particle.p = particle.get_pressure()

    # Обновление шага по времени, исходя из условия устойчивости
    def update_time_step(self):
        new_dt = 100.0
        dt_min = 100.0
        r_min = 100.0
        c_max = 0.0
        rh_max = 0.0
        for particle in self.particles:
            if particle.type != Type.FLEX: continue
            for neighbour in self.particles:
                if neighbour is particle: continue
                dr = particle.x - neighbour.x
                r = fabs(dr)
                h = 0.5 * (particle.h + neighbour.h)
                if r < 2.0 * h:
                    r_min = r if r < r_min else r_min
                    speed = 0.5 * (particle.get_sound_speed() + neighbour.get_sound_speed())
                    c_max = speed if c_max < speed else c_max
                    v = particle.v - neighbour.v
                    rh = dr * v
                    rh = h * rh / (r * r + self.c * h * h) if rh < 0.0 else 0.0
                    rh_max = rh if rh_max < rh else rh_max
            new_dt = r_min / (c_max * (1.0 + 1.2 * self.a) + 1.2 * self.b * rh_max)
            dt_min = new_dt if new_dt < dt_min else dt_min
        dt_min *= self.courant
        self.dt = dt_min if dt_min < self.dt else self.dt
        self.dt = self.border_dt if self.border_dt is not None and self.dt < self.border_dt else self.dt
        return self.dt


# Класс-контроллер (инициализирует данные и оперирует классом-решателем для осуществления моделирования)
class MainController:
    def __init__(self, start, end, dt, border_dt=None, gamma=1.4, kh=4.0/3.0, courant=0.6):
        self.start = start
        self.end = end
        self.dt = dt
        self.gamma = gamma
        self.kh = kh
        self.border_dt = border_dt
        self.courant = courant
        self.particles = []

    def create_particles(self, n, left, right, lock_layer, rho_r, rho_l, p_r, p_l, v_r, v_l):
        dx = fabs(right - left) / (n + 2 * lock_layer)
        for i in range(n + 2 * lock_layer):
            if i <= (n + 2 * lock_layer) / 2:
                v = v_l
                rho = rho_l
                p = p_l
            else:
                v = v_r
                rho = rho_r
                p = p_r
            if i < lock_layer or i >= n + lock_layer:
                type = Type.SOLID
                v = 0.0
            else:
                type = Type.FLEX
            self.particles.append(Particle(left + dx * i,
                                           v,
                                           dx * rho,
                                           rho,
                                           Particle.get_energy(p, rho, self.gamma),
                                           0,
                                           0,
                                           type,
                                           self.gamma,
                                           self.kh))

    # Процесс моделирования с использованием класса-решателя
    def calculate(self):
        t = 0.0
        solver = SPHSolver(self.particles, self.dt, self.border_dt, self.courant)
        while t < self.end:
            solver.compute_rho()
            solver.compute_forces()
            self.dt = solver.update_time_step()
            solver.predict()
            solver.compute_rho()
            solver.compute_forces()
            solver.correct()
            t += self.dt
            #print("dt = {0:f}, t = {1:f}".format(self.dt, t))
        self.particles = solver.get_particles()

    # Построение точного решения (эталонного) из файла
    def exact_solution(self, file, ax1, ax2, ax3, ax4):
        exact = []
        with open(file) as f:
            str = f.read().split("\n")
            for s in str:
                col = s.split(" ")
                exact.append(col)
        X = zeros([len(exact)])
        P = zeros([len(exact)])
        RHO = zeros([len(exact)])
        V = zeros([len(exact)])
        E = zeros([len(exact)])
        for i in range(0, len(exact)):
            X[i] = exact[i][0]
            RHO[i] = exact[i][1]
            V[i] = exact[i][2]
            P[i] = exact[i][3]
            E[i] = exact[i][4]
        ax1.plot(X, RHO, c="red", linewidth="2.5", linestyle="-", label="exact")
        ax2.plot(X, V, c="red", linewidth="2.5", linestyle="-", label="exact")
        ax3.plot(X, P, c="red", linewidth="2.5", linestyle="-", label="exact")
        ax4.plot(X, E, c="red", linewidth="2.5", linestyle="-", label="exact")
        return ax1, ax2, ax3, ax4

    # Построение графиков
    def make_plot(self, file, title, ax1, ax2, ax3, ax4):
        ax2.set(title=title)
        ax1.set_xlabel("X", fontsize=12)
        ax1.set_ylabel("Rho", fontsize=12)
        ax2.set_xlabel("X", fontsize=12)
        ax2.set_ylabel("Speed", fontsize=12)
        ax3.set_xlabel("X", fontsize=12)
        ax3.set_ylabel("Pressure", fontsize=12)
        ax4.set_xlabel("X", fontsize=12)
        ax4.set_ylabel("Energy", fontsize=12)
        ax1.set_aspect("auto")
        ax2.set_aspect("auto")
        ax3.set_aspect("auto")
        ax4.set_aspect("auto")
        ax1, ax2, ax3, ax4 = self.exact_solution(file, ax1, ax2, ax3, ax4)
        count = len(self.particles)
        X = zeros([count])
        P = zeros([count])
        RHO = zeros([count])
        V = zeros([count])
        E = zeros([count])
        i = 0
        for particle in self.particles:
            X[i] = particle.x
            P[i] = particle.p
            RHO[i] = particle.rho
            V[i] = particle.v
            E[i] = particle.energy
            i += 1
        ax1.plot(X, RHO, c="green", linewidth="2.0", linestyle="-", label="SPH {0:1}p".format(count))
        ax2.plot(X, V, c="green", linewidth="2.0", linestyle="-", label="SPH {0:1}p".format(count))
        ax3.plot(X, P, c="green", linewidth="2.0", linestyle="-", label="SPH {0:1}p".format(count))
        ax4.plot(X, E, c="green", linewidth="2.0", linestyle="-", label="SPH {0:1}p".format(count))
        ax1.legend(loc="best")
        ax2.legend(loc="best")
        ax3.legend(loc="best")
        ax4.legend(loc="best")
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        return ax1, ax2, ax3, ax4


if __name__ == "__main__":
    n = 100
    lock_layer = 5
    dt = 0.01
    left = -0.5
    right = 0.5
    start = 0.0

    fig = plt.figure(num="Задание №1: Молекулярная динамика (метод SPH)")
    ax1 = fig.add_subplot(3, 4, 1)
    ax2 = fig.add_subplot(3, 4, 2)
    ax3 = fig.add_subplot(3, 4, 3)
    ax4 = fig.add_subplot(3, 4, 4)
    ax5 = fig.add_subplot(3, 4, 5)
    ax6 = fig.add_subplot(3, 4, 6)
    ax7 = fig.add_subplot(3, 4, 7)
    ax8 = fig.add_subplot(3, 4, 8)
    ax9 = fig.add_subplot(3, 4, 9)
    ax10 = fig.add_subplot(3, 4, 10)
    ax11 = fig.add_subplot(3, 4, 11)
    ax12 = fig.add_subplot(3, 4, 12)
    fig.tight_layout()

    # 1) Ударная трубка Г.Сода
    left = 0.0
    right = 1.0
    end = 0.2
    rho_l = 1.0
    rho_r = 0.125
    p_l = 1.0
    p_r = 0.1
    v_l = 0.0
    v_r = 0.0
    control = MainController(start, end, dt)
    control.create_particles(n, left, right, lock_layer, rho_r, rho_l, p_r, p_l, v_r, v_l)
    start_time = time.time()
    control.calculate()
    exec_time = abs(time.time() - start_time)
    ax1, ax2, ax3, ax4 = control.make_plot("sod_test.txt", "Ударная трубка Г.Сода", ax1, ax2, ax3, ax4)
    print("sod_test_done t = {0:f}!".format(exec_time))

    # 2) Тест на сильную ударную волну (Торо)
    end = 0.012
    left = 0.0
    right = 1.0
    p_l = 1000.0
    p_r = 0.01
    v_l = 0.0
    v_r = 0.0
    rho_l = 1
    rho_r = 1
    control = MainController(start, end, dt)
    control.create_particles(n, left, right, lock_layer, rho_r, rho_l, p_r, p_l, v_r, v_l)
    start_time = time.time()
    control.calculate()
    exec_time = abs(time.time() - start_time)
    ax5, ax6, ax7, ax8 = control.make_plot("strong_shock_test.txt", "Шоковая волна Торо", ax5, ax6, ax7, ax8)
    print("strong_shock_test_done t = {0:f}!".format(exec_time))

    # 3) Двойное разрежение Торо
    end = 0.15
    left = 0.0
    right = 1.0
    rho_l = 1.0
    rho_r = 1.0
    v_l = -2.0
    v_r = 2.0
    p_l = 0.4
    p_r = 0.4
    border_dt = 0.0001
    control = MainController(start, end, dt, border_dt)
    control.create_particles(n, left, right, lock_layer, rho_r, rho_l, p_r, p_l, v_r, v_l)
    start_time = time.time()
    control.calculate()
    exec_time = abs(time.time() - start_time)
    ax9, ax10, ax11, ax12 = control.make_plot("double_rarefaction_test.txt", "Двойная волна разрежения Торо", ax9, ax10, ax11, ax12)
    print("double_rarefaction_test_done t = {0:f}!".format(exec_time))

    fig.set_figwidth(16)
    fig.set_figheight(9)
    plt.savefig("Result.png")
    plt.show()
