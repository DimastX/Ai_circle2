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
    'v0': 20.0,         # желаемая скорость (м/с)
    'delta': 4,       # показатель степени
    's0': 2.0,          # минимальный зазор
    'N': 125           # число машин
}

def ring_mean_s(params):
    """
    s_mean= (L_ring - N*L_car)/ N, если это >0, иначе 0
    """
    L_ring= params['L_ring']
    L_car= params['L_car']
    N= params['N']
    free_space= L_ring - N* L_car
    if free_space< 0:
        free_space= 0.0
    return free_space/ N

def eq_for_ve(ve, T, params):
    """
    0= a[1 - (ve/v0)^delta - ((s0 + T ve)/ s_mean)^2].
    """
    a= params['a']
    v0= params['v0']
    delta= params['delta']
    s0= params['s0']
    s_mean= ring_mean_s(params)

    lhs= 1.0 - (ve/ v0)**delta - ((s0+ T* ve)/ s_mean)**2
    return a* lhs  # хотим=0

def find_equilibrium_velocity(params, T_val):
    """
    Ищем ve>0, решая eq_for_ve(ve,T_val)=0 методом bisect.
    """
    left= 0.0
    right= params['v0']* 3.0
    for _ in range(100):
        mid= 0.5*(left+ right)
        f_left= eq_for_ve(left, T_val, params)
        f_mid= eq_for_ve(mid, T_val, params)

        if abs(f_mid)< 1e-7:
            return mid
        if f_left* f_mid<= 0.0:
            right= mid
        else:
            left= mid
    return 0.5*(left+ right)

def compute_A_and_2B(ve, T_val, params):
    """
    Считаем A и 2B (т.к. max real= A+2B при k= pi).
    """
    a= params['a']
    b= params['b']
    v0= params['v0']
    delta= params['delta']
    s0= params['s0']
    s_mean= ring_mean_s(params)

    # s^*(ve,0)= s0+ T_val* ve
    s_star_e= s0+ T_val* ve

    # s_e= s_mean (предполагаем)
    s_e= s_mean

    # A:
    part1= - a* delta*(ve/ v0)**(delta-1)/ v0
    part2= - 2*a*( s_star_e/(s_e**2 ))* T_val
    A= part1+ part2

    # B =>(A+2B) => first find B
    partB= - 2*a*( s_star_e/( s_e**2))*( -ve/(2.0* np.sqrt(a*b)) )
    B= partB
    return A, 2.0* B

def f_lambda(lam, alpha, tau_r):
    return -lam + alpha * np.exp(-lam * tau_r)

def solve_lambda_delay(alpha, tau_r, search_range=(-100, 100), steps=500):
    # Ищем отрезок, где f меняет знак
    z = alpha * tau_r
    solutions = []

    for branch in [0, -1]:  # W₀ и W₋₁
        w = lambertw(z, k=branch)
        if np.isreal(w):
            lam = w.real / tau_r
            solutions.append(lam)
    if len(solutions) == 0:
        return [np.nan]
    return solutions


def run_stability_vs_T(params, T_array, reaction_delay):
    results= []
    for T_val in T_array:
        ve= find_equilibrium_velocity(params, T_val)
        if ve< 1e-5:
            # нулевое решение => обычно считаем stable
            results.append( (T_val, ve, [0.0], True) )
            continue
        # Считаем A,2B
        A, twoB= compute_A_and_2B(ve, T_val, params)
        # => A+ 2B
        Aplus2B= A+ twoB
        # Решаем lambda= (A+2B) e^{-lambda tau_r}
        lam_list= solve_lambda_delay(Aplus2B, reaction_delay)
        # Если хотя бы одно lam>0 => неустойчиво
        stable= all(lam <= 0 for lam in lam_list)
        results.append( (T_val, ve, lam_list, not stable, Aplus2B) )
    return results

if __name__=='__main__':
    T_values= np.linspace(0.5, 5.0, 100)
    reaction_delay= 0.5   # 0.5 c

    data= run_stability_vs_T(base_params, T_values, reaction_delay)

    # Готовим графики
    x_vals= []
    lam_vals= []
    st_ind= []
    
    for d in data:
        T_val, ve, lam_list, unstbl, _ = d
        x_vals.extend([T_val] * len(lam_list))
        lam_vals.extend(lam_list)
        st_ind.extend([1.0 if (lam <= 0) else 0.0 for lam in lam_list])

    plt.figure(figsize=(8,5))
    ax1= plt.gca()
    color1='tab:red'
    ax1.set_xlabel('Headway T, с')
    ax1.set_ylabel('lambda (delay eq)', color=color1)
    ax1.plot(x_vals, lam_vals, 'o', color=color1, label='lambda')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    ax2= ax1.twinx()
    color2='tab:blue'
    ax2.set_ylabel('Устойчивость (1 - устойчиво, 0 - неустойчиво)', color=color2)
    ax2.plot(x_vals, st_ind, '-', color=color2, label='stable?')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.1,1.1)

    plt.title(f'Зависимость lambda от T, tau_r= {reaction_delay}s')
    plt.show()

    for row in data:
        T_val, ve, lam_list, unstbl, Aplus2B= row
        print(f"T= {T_val:.2f}, v_e= {ve:.3f}, lambda= {lam_list}, Unstable= {unstbl}, A+2B= {Aplus2B:.4f}")
