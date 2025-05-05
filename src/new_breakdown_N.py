import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.special import lambertw

base_params = {
    'L_ring': 3137.4,   # длина кольца (м)
    'L_car': 5.0,       # длина авто
    'a': 2.6,           # комфортное ускорение
    'b': 4.5,           # комфортное торможение
    'v_e': 20.0,        # равновесная скорость (м/с)
    'v0': 20.0,         # желаемая скорость (м/с)
    'delta': 4,         # показатель степени
    's0': 2.0,          # минимальный зазор
}

def ring_mean_s(N, params):
    """
    s_mean = L_ring/N - L_car
    """
    L_ring = params['L_ring']
    L_car = params['L_car']
    if N <= 0:
        return 1e-10  # маленькое положительное число вместо бесконечности
    free_space = L_ring/N - L_car
    if free_space < 0:
        return 1e-10  # маленькое положительное число вместо нуля
    return free_space

def eq_for_N(N, T, params):
    """
    При v_e = v0:
    0 = ((s0 + T v_e)/(L_ring/N - L_car))^2
    Решаем относительно N:
    N = L_ring/(L_car + s0 + T*v_e)
    """
    L_ring = params['L_ring']
    L_car = params['L_car']
    v_e = params['v_e']
    s0 = params['s0']
    
    # Вычисляем N напрямую
    N_calc = L_ring/(L_car + s0 + T*v_e)
    return N_calc

def find_equilibrium_N(params, T_val):
    """
    Находим N напрямую из уравнения
    """
    N = eq_for_N(1, T_val, params)  # первый аргумент не используется
    return N

def compute_A_and_2B(N, T_val, params):
    """
    Считаем A и 2B (т.к. max real = A+2B при k = pi)
    """
    a = params['a']
    b = params['b']
    v_e = params['v_e']
    delta = params['delta']
    s0 = params['s0']
    s_mean = ring_mean_s(N, params)
    
    if s_mean <= 0:
        return 0.0, 0.0
    
    # s^*(v_e,0) = s0 + T_val * v_e
    s_star_e = s0 + T_val * v_e
    
    # A:
    part1 = -a * delta * (v_e/params['v0'])**(delta-1)/params['v0']
    part2 = -2*a * (s_star_e/(s_mean**2)) * T_val
    A = part1 + part2
    
    # B => (A+2B) => first find B
    partB = -2*a * (s_star_e/(s_mean**2)) * (-v_e/(2.0 * np.sqrt(a*b)))
    B = partB
    return A, 2.0 * B

def run_stability_vs_T(params, T_array, reaction_delay):
    results = []
    for T_val in T_array:
        N = find_equilibrium_N(params, T_val)
        if N < 1:
            results.append((T_val, N, [0.0], True))
            continue
        
        # Считаем A,2B
        A, twoB = compute_A_and_2B(N, T_val, params)
        # => A + 2B
        Aplus2B = A + twoB
        # Решаем lambda = (A+2B) e^{-lambda tau_r}
        lam_list = solve_lambda_delay(Aplus2B, reaction_delay)
        # Если хотя бы одно lam > 0 => неустойчиво
        stable = all(lam <= 0 for lam in lam_list)
        results.append((T_val, N, lam_list, not stable, Aplus2B))
    return results

def solve_lambda_delay(alpha, tau_r, search_range=(-100, 100), steps=500):
    """
    Решаем уравнение lambda = alpha * e^(-lambda * tau_r)
    """
    if abs(alpha) > 1e10:
        return [float('inf')]
    elif abs(alpha) < 1e-10:
        return [0.0]
    
    z = alpha * tau_r
    solutions = []

    for branch in [0, -1]:  # W₀ и W₋₁
        try:
            w = lambertw(z, k=branch)
            if np.isreal(w):
                lam = w.real / tau_r
                solutions.append(lam)
        except:
            continue
            
    if len(solutions) == 0:
        return [0.0]
    return solutions

if __name__ == '__main__':
    T_values = np.linspace(0.5, 5.0, 100)
    reaction_delay = 0.5   # 0.5 c

    data = run_stability_vs_T(base_params, T_values, reaction_delay)

    # Готовим графики
    x_vals = []
    lam_vals = []
    st_ind = []
    N_vals = []
    
    for d in data:
        T_val, N, lam_list, unstbl, _ = d
        x_vals.extend([T_val] * len(lam_list))
        lam_vals.extend(lam_list)
        st_ind.extend([1.0 if (lam <= 0) else 0.0 for lam in lam_list])
        N_vals.append(N)

    plt.figure(figsize=(12,5))
    
    # Первый график: lambda и устойчивость
    ax1 = plt.subplot(121)
    color1 = 'tab:red'
    ax1.set_xlabel('Headway T, с')
    ax1.set_ylabel('lambda (delay eq)', color=color1)
    ax1.plot(x_vals, lam_vals, 'o', color=color1, label='lambda')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Устойчивость (1 - устойчиво, 0 - неустойчиво)', color=color2)
    ax2.plot(x_vals, st_ind, '-', color=color2, label='stable?')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.1,1.1)

    # Второй график: N vs T
    ax3 = plt.subplot(122)
    ax3.plot(T_values, N_vals, 'g-', label='N')
    ax3.set_xlabel('Headway T, с')
    ax3.set_ylabel('Количество машин N')
    ax3.grid(True)
    ax3.set_ylim(0, max(N_vals)*1.1)

    plt.suptitle(f'Зависимость lambda и N от T, tau_r= {reaction_delay}s')
    plt.tight_layout()
    plt.show()

    for row in data:
        T_val, N, lam_list, unstbl, Aplus2B = row
        print(f"T= {T_val:.2f}, N= {N:.1f}, lambda= {lam_list}, Unstable= {unstbl}, A+2B= {Aplus2B:.4f}") 