import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve, brentq, root
from run_straight import run_simulation
from generate_straight_rou import generate_straight_rou
import pandas as pd
from analyze_straight_data import analyze_straight_data

def equilibrium_velocity(T, q, s0, v0, delta, L, approx):
    # Решаем уравнение: (v_e/v0)^delta + [q(s0 + T v_e)/v_e]^2 = 1
    def func(v):
        return (v/v0)**delta + ((s0 + T*v)/(v/q - L))**2 - 1
    # Начальное приближение: v0*0.8
    v_e, = fsolve(func, v0*approx)
    return v_e

def analyze_experiment(results_dir, tau, q_fixed, s0, a, b, v0, delta, eidm_params):
    # Анализируем экспериментальные данные и считаем Re(lambda)
    density_flow_file = os.path.join(results_dir, 'density_flow_data.csv')
    if not os.path.exists(density_flow_file):
        print(f"Файл {density_flow_file} не найден!")
        return None, None
    df = pd.read_csv(density_flow_file)
    q_exp = df['flow'].mean()
    v_exp = df['mean_speed'].mean()
    rho_exp = df['density'].mean()
    if np.isnan(rho_exp) or np.isnan(q_exp) or np.isnan(v_exp):
        print("Ошибка анализа данных: NaN")
        return None, None
    # Восстанавливаем T_exp по формуле: T = (1/rho - s0)/v_exp
    T_exp = (1/rho_exp - s0)/v_exp
    S_e_exp = 1/rho_exp
    s_e_exp = S_e_exp - eidm_params['length_mean']
    v_e_exp = v_exp
    # Коэффициенты
    term1 = (v_e_exp/v0)**delta
    A = -a * (delta/v_e_exp*term1 + 2*T_exp*(s0 + T_exp*v_e_exp)/s_e_exp**2)
    B = a * v_e_exp * (s0 + T_exp*v_e_exp) / (np.sqrt(a*b) * s_e_exp**2)
    C = 2 * a * (s0 + T_exp*v_e_exp)**2 / s_e_exp**3
    D = A + 2*B
    omega = np.sqrt(0.5*(D**2 + np.sqrt(D**4 + 16*C**2)))
    phi = omega * tau
    Re_lambda = D * np.cos(phi) - omega * np.sin(phi)
    return T_exp, Re_lambda

def lambda_max_formula2(T, q, s0, L, a, b, delta, k_grid, v_e):
    # formulas2.md: v0→∞
    # v_e = q * (s0 + L) / (1 - q * T)
    s_e = v_e / q - L
    A = 2 * a / s_e
    B = 2 * a * T / s_e
    C = a * v_e / (np.sqrt(a * b) * s_e)
    lambdas = []
    for k in k_grid:
        exp_jk = np.exp(-1j * k)
        # Характеристическое уравнение: λ^2 - λ[B + C(1-exp_jk)] - A(exp_jk-1) = 0
        coeffs = [1, -(B + C * (1 - exp_jk)), -A * (exp_jk - 1)]
        roots = np.roots(coeffs)
        lambdas.extend(roots)
    # Возвращаем максимум Re(λ)
    return np.max(np.real(lambdas))

def find_all_roots(func, v_min, v_max, num=200, **kwargs):
    """
    Ищет все корни функции func на интервале [v_min, v_max] методом смены знака и brentq.
    Возвращает список корней.
    """
    vs = np.linspace(v_min, v_max, num)
    roots = []
    for i in range(len(vs)-1):
        v1, v2 = vs[i], vs[i+1]
        f1, f2 = func(v1), func(v2)
        if np.isnan(f1) or np.isnan(f2):
            continue
        if f1 * f2 < 0:
            try:
                root = brentq(func, v1, v2, **kwargs)
                # Проверяем, что корень не повторяется (с учётом точности)
                if not any(np.isclose(root, r, atol=1e-6) for r in roots):
                    roots.append(root)
            except Exception as e:
                print(f"    [find_all_roots] Ошибка поиска корня на [{v1:.4f}, {v2:.4f}]: {e}")
    return roots

def lambda_max_formula3(T, q, s0, L, v0, a, b, delta, k_grid, tau=0.5):
    # formulas3.md: полный E-IDM, численное решение с учётом времени реакции tau
    from scipy.optimize import root
    def G(v):
        denom = v / q - L
        if denom <= 0:
            return 1e6  # вне допустимой области
        return 1 - (v / v0) ** delta - ((s0 + T * v) / denom) ** 2
    v_min = q * L + 1e-6
    v_max = v0
    roots = find_all_roots(G, v_min, v_max, num=300)
    phys_roots = [v for v in roots if v > 0 and v < v0 and (v/q - L) > 0]
    if not phys_roots:
        # print(f"    [formula3] Нет физических корней!")
        return np.nan
    v_e = min(phys_roots)
    s_e = v_e / q - L
    A = 2 * a / s_e
    B = a * (delta * (v_e / v0) ** (delta - 1) / v0 + 2 * T / s_e)
    C = a * v_e / (np.sqrt(a * b) * s_e)
    lambdas_max = []
    for k in k_grid:
        exp_jk = np.exp(-1j * k)
        # Квадратное уравнение (tau=0) для начальных guess
        coeffs = [1, -(B + C * (1 - exp_jk)), -A * (exp_jk - 1)]
        quad_roots = np.roots(coeffs)
        # Для каждого корня решаем transcendental уравнение
        found_lambdas = []
        for guess in quad_roots:
            def F(lam):
                lam_c = lam[0] + 1j*lam[1]
                val = lam_c**2 * np.exp(lam_c * tau) - lam_c * (B + C * (1 - exp_jk)) - A * (exp_jk - 1)
                return [val.real, val.imag]
            try:
                sol = root(F, [guess.real, guess.imag], method='hybr', tol=1e-8, options={'maxfev': 50})
                if sol.success:
                    lam_c = sol.x[0] + 1j*sol.x[1]
                    # Проверяем, что корень не повторяется
                    if not any(np.isclose(lam_c, l, atol=1e-6) for l in found_lambdas):
                        found_lambdas.append(lam_c)
            except Exception as e:
                print(f"    [formula3] Ошибка root по lambda: {e}")
        if found_lambdas:
            max_re = np.max([l.real for l in found_lambdas])
            lambdas_max.append(max_re)
        else:
            lambdas_max.append(np.nan)
    # Возвращаем максимум по всем k
    return np.nanmax(lambdas_max)

def main():
    # Фиксированные параметры
    step_length = 0.5  # Это tau (driver reaction delay) в формулах
    N = 100
    q_fixed = 0.25
    s0 = 2.0
    v0 = 25
    a = 2.6
    b = 4.5
    delta = 4.0
    stepping = 0.5
    L = 5
    # Диапазон для tau_eidm (это T в формулах)
    tau_eidm_vals = np.linspace(0.7, 3.9, 10)
    tau_eidm_analytic_vals = np.linspace(0.7, 1.2, 100)
    k_grid = np.linspace(1e-4, np.pi, 300)  # сетка по k для поиска максимума
    Re_lambda_exp_list = []
    T_exp_list = []
    Re_lambda_analytic = []
    Re_lambda_exp_vs_tau_eidm = []
    Re_lambda_formula2 = []
    Re_lambda_formula3 = []
    D_analytic = []
    D_exp_vs_Tsweep = []
    # # Аналитика (гладкая кривая)
    for tau_eidm in tau_eidm_analytic_vals:
        v_e = equilibrium_velocity(tau_eidm, q_fixed, s0, v0, delta, L, 0.4)
        s_e = v_e / q_fixed - L
        term1 = (v_e/v0)**(delta-1)
        A = -a * (delta * term1 /v0 + 2*tau_eidm*(s0 + tau_eidm*v_e)/s_e**2)
        B = a * v_e * (s0 + tau_eidm*v_e) / (np.sqrt(a*b) * s_e**2)
        C = 2 * a * (s0 + tau_eidm*v_e)**2 / s_e**3 
        D = A + 2*B
        omega = np.sqrt(0.5*(D**2 + np.sqrt(D**4 + 16*C**2)))
        phi = omega * step_length
        Re_lambda = D * np.cos(phi) - omega * np.sin(phi)
        D_analytic.append(D)
        Re_lambda_analytic.append(Re_lambda)
        # Новый расчёт по formulas2.md
        Re_lambda_formula2.append(lambda_max_formula2(tau_eidm, q_fixed, s0, L, a, b, delta, k_grid, v_e))
        # Новый расчёт по formulas3.md (теперь с tau=step_length)
        Re_lambda_formula3.append(lambda_max_formula3(tau_eidm, q_fixed, s0, L, v0, a, b, delta, k_grid, tau=step_length))
        # print(f"tau_eidm: {tau_eidm:.4f}, Re_lambda: {Re_lambda:.4f}, formula2: {Re_lambda_formula2[-1]:.4f}, formula3: {Re_lambda_formula3[-1]:.4f}")
    # Эксперимент (редкие точки)
    for tau_eidm in tau_eidm_vals:
        # 1. Вычисляем v_e и s_e
        v_e = equilibrium_velocity(tau_eidm, q_fixed, s0, v0, delta, L, 0.4)
        G = lambda v: 1 - (v/v0)**delta - ((s0 + tau_eidm*v)/(v/q_fixed - L))**2
        print(find_all_roots(G, v_e, v0, num=300))
        # v_e = q_fixed * (s0 + L) / (1 - q_fixed * tau_eidm)
        s_e = (s0 + tau_eidm*v_e)/np.sqrt(1-(v_e/v0)**delta)
        term1 = (v_e/v0)**(delta-1)
        A = -a * (delta * term1 /v0 + 2*tau_eidm*(s0 + tau_eidm*v_e)/s_e**2)
        print(f"A: {A:.4f}")
        B = a * v_e * (s0 + tau_eidm*v_e) / (np.sqrt(a*b) * s_e**2)
        print(f"B: {B:.4f}")
        C = 2 * a * (s0 + tau_eidm*v_e)**2 / s_e**3
        print(f"C: {C:.4f}")
        D = A + 2*B
        D_exp_vs_Tsweep.append(D)
        print(f"D: {D:.4f}")
        print(f"s_e: {s_e:.4f}")
        s_e = s0 + tau_eidm * v_e
        print(f"s_e: {s_e:.4f}")
        s_e = v_e / q_fixed - L
        print(f"s_e: {s_e:.4f}")
        # --- Выводим параметры в консоль ---
        print(f"\n=== Параметры для tau_eidm (T) = {tau_eidm:.4f} ===")
        print(f"s_0 (min_gap): {s0:.4f}")
        print(f"step_length (tau): {step_length:.4f}")
        print(f"q (fixed): {q_fixed:.4f}")
        print(f"v_e (равновесная скорость): {v_e:.4f}")
        print(f"s_e (равновесный интервал): {s_e:.4f}")
        # 2. Формируем параметры EIDM для текущей симуляции
        eidm_params = {
            'accel_mean': a,
            'accel_std': 0,
            'decel_mean': b,
            'decel_std': 0,
            'sigma_mean': 0,
            'sigma_std': 0,
            'tau_mean': tau_eidm,  # tau_eidm = T (time gap)
            'tau_std': 0,
            'delta_mean': delta,
            'delta_std': 0,
            'stepping_mean': step_length,
            'stepping_std': 0,
            'length_mean': L,
            'length_std': 0,
            'min_gap_mean': s0,
            'min_gap_std': 0,
            'max_speed_mean': v0,
            'max_speed_std': 0.0
        }
        # 3. Генерируем rou.xml с этими параметрами
        generate_straight_rou(N, eidm_params, q_fixed, start_speed=v_e, equilibrium_spacing=s_e)
        # 4. Запускаем симуляцию (step_length теперь явно передаём)
        results_dir = run_simulation(N, eidm_params, q_fixed, step_length=step_length, v_e=v_e)
        # 5. Анализируем экспериментальные данные
        T_exp, Re_lambda_exp = analyze_experiment(results_dir, step_length, q_fixed, s0, a, b, v0, delta, eidm_params)
        if T_exp is not None and Re_lambda_exp is not None:
            T_exp_list.append(T_exp)
            Re_lambda_exp_list.append(Re_lambda_exp)
            Re_lambda_exp_vs_tau_eidm.append(Re_lambda_exp)  # y=Re_lambda_exp, x=tau_eidm
        else:
            Re_lambda_exp_vs_tau_eidm.append(np.nan)
        # 6. Запуск анализа траекторий и построения графиков
        sim_data_file = os.path.join(results_dir, 'simulation_data.csv')
        if os.path.exists(sim_data_file):
            analyze_straight_data(sim_data_file)
    # 8. Строим итоговый график
    plt.figure(figsize=(8,6))
    plt.plot(tau_eidm_analytic_vals, Re_lambda_analytic, label='Аналитика (основная)', color='blue')
    plt.plot(tau_eidm_analytic_vals, Re_lambda_formula2, label='E-IDM v0→∞ (formulas2.md)', color='orange', linestyle='--')
    plt.plot(tau_eidm_analytic_vals, Re_lambda_formula3, label='E-IDM полный (formulas3.md)', color='purple', linestyle='-.')
    plt.scatter(T_exp_list, Re_lambda_exp_list, color='red', label='Эксперимент (SUMO, x=T_exp)')
    plt.scatter(tau_eidm_vals, Re_lambda_exp_vs_tau_eidm, color='green', marker='x', label='Эксперимент (SUMO, x=T из sweep)')
    for x, y in zip(T_exp_list, Re_lambda_exp_list):
        plt.annotate(f"T={x:.2f}\nReλ={y:.2f}", (x, y), textcoords="offset points", xytext=(10,10), ha='left', color='red')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('T (tau_eidm, сек)')
    plt.ylabel('Re(lambda)')
    plt.title('Устойчивость потока: Re(lambda) от T (tau_eidm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('stability_vs_T.png')
    plt.show()
    # === Новый график D(T) ===
    plt.figure(figsize=(8,6))
    plt.plot(tau_eidm_vals, D_exp_vs_Tsweep, 'gx-', label='Эксперимент D(T) (x=T из sweep)')
    plt.plot(T_exp_list, D_exp_vs_Tsweep, 'ro', label='Эксперимент D(T) (x=T_exp)')
    plt.xlabel('T (сек)')
    plt.ylabel('D = A + 2B')
    plt.title('Зависимость D от T')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('D_vs_T.png')
    plt.show()

if __name__ == "__main__":
    main() 