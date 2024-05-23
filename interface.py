import tkinter as tk
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox as mb


def run_program():
    dump_params = {}
    for param_name, var in params.items():
        dump_params[param_name] = var.get()

    with open('beam_conf.json', 'w') as f:
        json.dump(dump_params, f)
    
    result = subprocess.run(['./cmake-build-debug/SchrodingerEquation'], capture_output=True, text=True)
    # Проверка результата выполнения команды
    print(result.returncode)  # Вывод кода возврата внешней программы
    print(result.stdout)      # Вывод стандартного потока вывода внешней программы
    print(result.stderr)      # Вывод стандартного потока ошибок внешней программы

    data = np.loadtxt('solution_cuda.txt', usecols=(0, 1, 2)) 
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]

    X = np.unique(X)
    Y = np.unique(Y)
    X, Y = np.meshgrid(X, Y)
    Z = Z.reshape(X.shape)

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, cmap='viridis')  
    cbar = fig.colorbar(contour)  
    cbar.ax.set_ylabel('$\\rho/|a_0|^2$')
    ax.set_xlabel('$\\zeta$')
    ax.set_ylabel('$\\tau$')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=0, column=2, rowspan=14, padx=10, sticky='nesw')
    canvas.draw()

def show_about_window():
    about_text = "N - кол-во частиц SPH.\n" \
        "DT - шаг по времени.\n" \
        "NT - общее кол-во шагов по времени.\n" \
        "zeta start - левая граница координаты zeta.\n" \
        "zeta end - правая граница координаты zeta.\n" \
        "kappa - kappa=kc^2/omega_0^2.\n" \
        "a - амплитуда элек-го поля пучка в нач. момент времени.\n" \
        "b - полуширина пучка в нач. момент времени.\n" \
        "m - тип нанотрубок.\n" \
        "alpha, L - макс. значения счётчиков alpha и l при расчёте нелинейности.\n" \
        "R=Q=-D - параметры, характеризующие нелинейные взаимодействия.\n" 
    mb.showinfo("Справка", about_text) 

# Создание главного окна
root = tk.Tk()
root.title("Моделирование распространения лазерного пучка в УНТ")

# Определение параметров
params = {
    "N": tk.IntVar(root, 500),
    "DT": tk.DoubleVar(root, 0.01),
    "NT": tk.IntVar(root, 3000),
    "zeta start": tk.DoubleVar(root, -10.0),
    "zeta end": tk.DoubleVar(root, 10.0),
    "kappa": tk.DoubleVar(root, 20),
    "a": tk.DoubleVar(root, 0.323),
    "b": tk.DoubleVar(root, 2.0),
    "m": tk.IntVar(root, 7),
    "alpha": tk.IntVar(root, 9),
    "L": tk.IntVar(root, 9),
    "R": tk.DoubleVar(root, 0.0),
    "Q": tk.DoubleVar(root, 0.0),
    "D": tk.DoubleVar(root, 0.0),
}

# Создание фрейма для графика
graph_frame = tk.Frame(root)
graph_frame.grid(row=0, column=3, padx=10, pady=10, sticky='nesw')

# Создание виджетов для параметров
i = 0
for param_name, var in params.items():
    label = tk.Label(root, text=f"{param_name}: ")
    entry = tk.Entry(root, textvariable=var)

    label.grid(row=i, column=0, sticky='w')
    entry.grid(row=i, column=1, sticky='ew')
    i=i+1

run_button = tk.Button(root, text="Рассчитать", command=run_program)
run_button.grid(row=len(params), column=1, sticky='e')

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=2, rowspan=14, padx=10, sticky='nesw')
canvas.draw()

main_menu = Menu(root, tearoff=0)
root.config(menu=main_menu)
help_menu = Menu(main_menu, tearoff=0)    
main_menu.add_command(label="Справка", command=show_about_window)

# Запуск главного окна
root.mainloop()