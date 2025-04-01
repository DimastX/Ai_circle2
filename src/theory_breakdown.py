"""
extended_stability.py

Скрипт, в котором:
1) Учитывается длина кольца L_ring;
2) Для каждой пары (N,T) решается уравнение равновесия:
   (v_e / v0)^delta + [ (s0 + T v_e) / (L_ring / N) ]^2 = 1,
   чтобы найти v_e;
3) Вычисляются A,B,C, потом ищется max Re(lambda) по k>0;
4) Рисуется график зависимости.

Зависимые параметры (при желании) можно изменять.
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath
from math import sqrt

def solve_equilibrium(v0, delta, a, b, T, s0, L_ring, L_car, N):
    """
    Решаем:
      (v_e / v0)^delta + [ (s0 + T v_e)/ s_e]^2 = 1
    где s_e= (L_ring - N*L_car)/ N.
    (Если (N*L_car) >= L_ring, s_e <=0 => нет решения)
    Используем дихотомию по v_e.
    """

    # 1) Считаем s_e
    s_e = (L_ring - N*L_car)/ N
    if s_e <= 0:
        # Физически нет места
        return None, None

    # 2) Решаeм (v_e / v0)^δ + [ (s0 + T v_e)/ s_e]^2=1
    left, right = 0.0, 1.5*v0
    eps = 1e-4
    max_iter = 1000
    iter_count = 0

    while (right-left) > eps and iter_count < max_iter:
        mid = 0.5*(left+ right)
        eq_val = (mid/ v0)**delta + ((s0+ T*mid)/ s_e)**2
        if eq_val > 1.0:
            right = mid
        else:
            left = mid
        iter_count += 1
    
    v_e = 0.5*(left+ right)

    # Проверяем точно ли eq_val близко к 1
    check_val = (v_e/ v0)**delta + ((s0+ T*v_e)/ s_e)**2
    if abs(check_val - 1.0) > 0.05:
        return None, None

    return v_e, s_e


def calc_sstar(v, dv, T, s0, a, b):
    """
    s*(v, dv) = s0 + T v + v(- dv)/(2 sqrt(a b))
    """
    return s0 + T*v + v*(-dv)/(2.0*sqrt(a*b))

def calc_A(v_e,v0,delta,a,T,s0,s_e):
    """
    A = ∂F/∂v при (v=v_e, s=s_e, Δv=0).
    F = a [ 1-(v/v0)^delta - (s^*/ s)^2 ]
    s^*(v, 0)= s0 + T v
    """
    s_star = s0 + T*v_e
    term1 = - delta * (v_e/v0)**(delta-1) / v0
    term2 = -2.0 * (s_star/(s_e**2)) * T
    return a*(term1 + term2)

def calc_B(v_e,v0,delta,a,b,T,s0,s_e):
    """
    B = ∂F/∂(Δv) при (v=v_e, s=s_e, Δv=0)
    B= a [ -2(s^*/ s^2)(∂ s^*/ ∂(Δv)) ]
    """
    s_star= s0 + T*v_e
    ds_dDv= - v_e/(2.0*sqrt(a*b))
    return a * (-2.0 * (s_star/(s_e**2))* ds_dDv)

def calc_C():
    """
    В классической IDM/EIDM C=0, 
    так как s^* не зависит явно от s.
    """
    return 0.0

def solve_lambda(k, A, B, C):
    """
    Решает квадр. уравнение: λ^2 = λ[A + B(1 - e^{-i k}) ] + C(e^{-i k}-1).
    """
    e_negik= cmath.exp(-1j*k)
    # перепишем: λ^2 - λ [ A + B(1- e^{-i k}) ] - C( e^{-i k}-1 )=0
    p= - (A+ B*(1-e_negik))  # при λ^2 + pλ + q=0
    q= - C*(e_negik-1)
    disc= p**2 - 4.0*q
    lam1= (-p+ cmath.sqrt(disc))/ 2.0
    lam2= (-p- cmath.sqrt(disc))/ 2.0
    return lam1, lam2

def compute_max_re_lambda(N, A, B, C):
    """
    Ищем max Re(λ) для k= 2π m/N, m=1..N-1.
    """
    best_k= None
    best_re= -1e9
    for m in range(1, N):
        k_val= 2.0*np.pi*m/N
        lam1, lam2= solve_lambda(k_val, A,B,C)
        re1, re2= lam1.real, lam2.real
        if re1> best_re:
            best_re= re1
            best_k= k_val
        if re2> best_re:
            best_re= re2
            best_k= k_val
    return best_k, best_re

def main(param_to_vary='N', param_range=None):
    """
    Анализ системы с возможностью выбора параметра для варьирования.
    
    :param param_to_vary: параметр для варьирования ('N', 'L_ring', 'a', 'b', 'v0', 'delta', 's0', 'T')
    :param param_range: диапазон значений для выбранного параметра (если None, используется значение по умолчанию)
    """
    # Базовые параметры
    base_params = {
        'L_ring': 3137.4,
        'L_car': 5.0,
        'a': 2.6,
        'b': 4.5,
        'v0': 20.0,
        'delta': 0.5,
        's0': 2.0,
        'T': 2,
        'N': 144  # базовое значение N
    }
    
    # Если диапазон не указан, используем значения по умолчанию
    if param_range is None:
        if param_to_vary == 'N':
            param_range = np.arange(36, 144, 1)
        elif param_to_vary == 'L_ring':
            param_range = np.linspace(2000, 4000, 50)
        elif param_to_vary == 'a':
            param_range = np.linspace(1.0, 4.0, 50)
        elif param_to_vary == 'b':
            param_range = np.linspace(2.0, 6.0, 50)
        elif param_to_vary == 'v0':
            param_range = np.linspace(15.0, 25.0, 50)
        elif param_to_vary == 'delta':
            param_range = np.linspace(0.3, 0.7, 50)
        elif param_to_vary == 's0':
            param_range = np.linspace(1.0, 3.0, 50)
        elif param_to_vary == 'T':
            param_range = np.linspace(0.3, 0.7, 50)
        else:
            raise ValueError(f"Неизвестный параметр для варьирования: {param_to_vary}")
    
    re_lambda_list = []
    
    for param_value in param_range:
        # Создаем копию базовых параметров и меняем нужный параметр
        current_params = base_params.copy()
        current_params[param_to_vary] = param_value
        
        # 1) Решаем уравнение равновесия => v_e
        v_e, s_e = solve_equilibrium(
            current_params['v0'], 
            current_params['delta'], 
            current_params['a'],
            current_params['b'], 
            current_params['T'], 
            current_params['s0'], 
            current_params['L_ring'],
            current_params['L_car'],  
            current_params['N']
        )
        
        # Проверяем, нашли ли мы решение
        if v_e is None or s_e is None:
            print(f"Предупреждение: не найдено решение для {param_to_vary} = {param_value}")
            continue
        
        # 3) находим A,B,C
        A_val = calc_A(v_e, current_params['v0'], current_params['delta'], 
                      current_params['a'], current_params['T'], 
                      current_params['s0'], s_e)
        B_val = calc_B(v_e, current_params['v0'], current_params['delta'], 
                      current_params['a'], current_params['b'], 
                      current_params['T'], current_params['s0'], s_e)
        C_val = calc_C()
        
        # 4) max Re(λ)
        best_k, best_re = compute_max_re_lambda(current_params['N'], A_val, B_val, C_val)
        re_lambda_list.append(best_re)
    
    # Рисуем
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, re_lambda_list, 'o-', label=f'max Re(λ) vs {param_to_vary}')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(f'{param_to_vary}')
    plt.ylabel('max Re(λ) among all k>0')
    plt.title(f'Зависимость max Re(λ) от {param_to_vary}')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Пример использования:
    # main('N')  # варьирование N
    main('T', np.linspace(0.7, 2, 50))  # варьирование T с заданным диапазоном
    # main('s0', np.linspace(2.0, 4.0, 200))  # варьирование s0 от 2.0 до 4.0
