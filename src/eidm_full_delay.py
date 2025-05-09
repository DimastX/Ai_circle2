#!/usr/bin/env python3
"""
E-IDM Stability Toolkit v2.2  —  с исправлениями по замечаниям
--------------------------------------------------------------
* Исправлен знак в B
* Уточнены коэффициенты кубического уравнения с Padé(1,1)
* Выводятся все три корня λ(k), отсортированные по Re
* Скан T c шагом 0.05 с
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import math, cmath, os, json

# ==== 1. Равновесие ====
def equilibrium(v0, delta, s0, T, L, q):
    def f(v):
        s = v/q - L
        if s <= 0:
            return 1.0
        return (v/v0)**delta + ((s0 + T*v)/s)**2 - 1
    xs = np.linspace(0.1, v0*1.3, 6000)
    signs = np.sign([f(x) for x in xs])
    roots = []
    for i in range(len(xs)-1):
        if signs[i]*signs[i+1] < 0:
            try:
                root = brentq(f, xs[i], xs[i+1])
                # добавляем только положительные и уникальные
                if root > 0 and all(abs(root - r) > 1e-3 for r in roots):
                    roots.append(root)
            except ValueError:
                pass
    return sorted(roots, reverse=True)

# ==== 2. Коэффициенты A, B, C ====
def coeffs(v_e, p, q):
    a, b = p['a'], p['b']
    v0, delta, s0, T, L = p['v0'], p['delta'], p['s0'], p['T'], p['L']
    s_e = v_e/q - L
    k2  = 1 - (v_e/v0)**delta
    A = 2*a * k2 / s_e
    # B — знак «–» исправлен
    B = -a * ( delta*(v_e/v0)**(delta-1)/v0
              + 2*T*k2/s_e )
    C =  a * v_e * k2 / ( math.sqrt(a*b) * s_e )
    return A, B, C

# ==== 3. Корни λ(k) с задержкой Padé(1,1) ====
def lambdas_delay(k, A, B, C, tau):
    z = cmath.exp(-1j*k)
    # верные коэффициенты из раскрытого Padé:
    # λ³
    p3 = 1.0
    # λ²: B + C(1−z) + 2/τ
    p2 = B + C*(1 - z) + 2.0/tau
    # λ¹: 2B/τ + 2C(1−z)/τ − A(1−z)
    p1 = 2.0*B/tau + 2.0*C*(1 - z)/tau - A*(1 - z)
    # λ⁰:  2A(1−z)/τ
    p0 = 2.0*A*(1 - z)/tau
    roots = np.roots([p3, p2, p1, p0])
    # сортируем по убыванию Re(λ)
    return sorted(roots, key=lambda x: x.real, reverse=True)

# ==== 4. Передаточная функция G(s) с Padé ====
def G_s(s, k, A, B, C, tau):
    z   = cmath.exp(-1j*k)
    num = 1 - s*tau/2
    den = (1 + s*tau/2)*(s**2) - num*(s*(B + C*(1 - z)) + A*(z-1))
    return num/den

# ==== 5. Анализ одного корня v_e ====
def analyse_root(idx, v_e, p, q, tau, figdir):
    A, B, C = coeffs(v_e, p, q)
    ks = np.linspace(1e-3, math.pi, 500)
    lam_re = [lambdas_delay(k, A, B, C, tau) for k in ks]
    # построим для двух наиболее медленных λ1, λ2
    l0 = [roots[0].real for roots in lam_re]   # самый медленный (наиб. Re)
    l1 = [roots[1].real for roots in lam_re]
    l2 = [roots[2].real for roots in lam_re]   # быстрый затухающий (обычно самый резко отрицательный)
    # Dispersion
    plt.figure()
    plt.plot(ks, l0, label='Re λ₀')
    plt.plot(ks, l1, label='Re λ₁')
    plt.plot(ks, l2, label='Re λ₂')
    plt.axhline(0, linestyle='--', color='k')
    plt.title(f'Dispersion root{idx}: v_e={v_e:.2f} m/s')
    plt.xlabel('k')
    plt.ylabel('Re λ')
    plt.legend(); plt.grid(True)
    plt.savefig(f'{figdir}/dispersion_root{idx}.png')
    plt.close()

    # критическое k по самому «медленному» корню λ₁
    re1 = np.array(l1)
    kcrit = ks[np.argmax(re1)]

    # Bode на kcrit (включая фазу)
    freqs = np.logspace(-2, 1.5, 400)
    G = np.array([G_s(1j*w, kcrit, A, B, C, tau) for w in freqs])
    plt.figure()
    plt.semilogx(freqs, 20*np.log10(np.abs(G)))
    plt.title(f'Bode mag root{idx}, k={kcrit:.2f}')
    plt.xlabel('ω'); plt.ylabel('|G| (dB)')
    plt.grid(True)
    plt.savefig(f'{figdir}/bode_mag_root{idx}.png')
    plt.close()

    plt.figure()
    plt.semilogx(freqs, np.angle(G, deg=True))
    plt.title(f'Bode phase root{idx}, k={kcrit:.2f}')
    plt.xlabel('ω'); plt.ylabel('Phase (°)')
    plt.grid(True)
    plt.savefig(f'{figdir}/bode_phase_root{idx}.png')
    plt.close()

    # Nyquist
    G_full = np.concatenate([G, np.conjugate(G[::-1])])
    plt.figure()
    plt.plot(G_full.real, G_full.imag)
    plt.axhline(0, color='k'); plt.axvline(0, color='k')
    plt.title(f'Nyquist root{idx}')
    plt.xlabel('Re'); plt.ylabel('Im')
    plt.axis('equal'); plt.grid(True)
    plt.savefig(f'{figdir}/nyquist_root{idx}.png')
    plt.close()

# ==== 6. Сканирование параметра T ====
def scan_T(p, q, tau, figdir):
    Ts = np.arange(0, 5, 0.05)  # шаг 0.05
    all_vals = {}
    v_e_legend_map = {}
    y_quantity_vals = {} # <<< НОВЫЙ СЛОВАРЬ ДЛЯ Y(T)

    for T_idx, T in enumerate(Ts):
        p_current = p.copy()
        p_current['T'] = T
        roots_v_e = equilibrium(p_current['v0'], p_current['delta'], 
                                p_current['s0'], p_current['T'], 
                                p_current['L'], q)

        if not roots_v_e:
            for root_label in all_vals.keys():
                if len(all_vals[root_label]) == T_idx:
                     all_vals[root_label].append(np.nan)
            # Добавляем nan и для нового словаря
            for root_label in y_quantity_vals.keys():
                if len(y_quantity_vals[root_label]) == T_idx:
                    y_quantity_vals[root_label].append(np.nan)
            continue

        for root_idx, v_e in enumerate(roots_v_e):
            N_ROOT_BRANCHES_TO_PLOT = 3 
            if root_idx >= N_ROOT_BRANCHES_TO_PLOT:
                continue

            root_label = f"v_e_branch_{root_idx}"
            
            # Инициализация/выравнивание для all_vals
            if root_label not in all_vals:
                all_vals[root_label] = [np.nan] * T_idx
                v_e_legend_map[root_label] = []
            if len(all_vals[root_label]) < T_idx:
                 all_vals[root_label].extend([np.nan] * (T_idx - len(all_vals[root_label])))
            elif len(all_vals[root_label]) > T_idx:
                 all_vals[root_label] = all_vals[root_label][:T_idx]

            # Инициализация/выравнивание для y_quantity_vals
            if root_label not in y_quantity_vals:
                y_quantity_vals[root_label] = [np.nan] * T_idx
            if len(y_quantity_vals[root_label]) < T_idx:
                 y_quantity_vals[root_label].extend([np.nan] * (T_idx - len(y_quantity_vals[root_label])))
            elif len(y_quantity_vals[root_label]) > T_idx:
                 y_quantity_vals[root_label] = y_quantity_vals[root_label][:T_idx]

            A, B, C = coeffs(v_e, p_current, q)
            # ОТЛАДОЧНЫЙ ВЫВОД
            # print(f"DEBUG: T={T:.2f}, v_e={v_e:.3f}, q={q:.3f}, L={p_current['L']:.1f}")
            s_e_calc_in_scan = v_e/q - p_current['L'] # Это s_e, как оно считается в coeffs
            # print(f"DEBUG: s_e_calc_in_scan = {s_e_calc_in_scan:.3f}")
            # print(f"DEBUG: A={A:.3e}, B={B:.3e}, C={C:.3e}")
            denom_calc_in_scan = 0.5 * A + B - C
            # print(f"DEBUG: Denominator (0.5A+B-C) = {denom_calc_in_scan:.3e}")
            if not np.isnan(denom_calc_in_scan) and abs(denom_calc_in_scan) < 1e-9:
                # print(f"DEBUG: Denominator is VERY CLOSE TO ZERO for T={T:.2f}, v_e={v_e:.3f}")
                pass
            # КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА

            re_max = -1e9
            y_val = np.nan # Значение по умолчанию для Y(T)

            # Проверка на NaN/Inf в A, B, C
            if any(map(lambda x: np.isnan(x) or np.isinf(x), [A, B, C])):
                re_max = np.nan # Уже присвоено выше для y_val
            else:
                # Расчет Re(lambda)
                for k_val in np.linspace(1e-3, math.pi, 300):
                    try:
                        current_lambdas = lambdas_delay(k_val, A, B, C, tau)
                        for lam in current_lambdas:
                            re_max = max(re_max, lam.real)
                    except np.linalg.LinAlgError:
                        pass # Пропускаем этот k, если roots не найдены
                
                # Расчет Y(T)
                denom = 0.5 * A + B - C
                if abs(denom) > 1e-9: # Проверка деления на ноль
                    y_val = 1.0 / denom
                else:
                    y_val = np.nan # Или можно np.inf, но nan безопаснее для графиков

            all_vals[root_label].append(re_max if re_max != -1e9 else np.nan)
            y_quantity_vals[root_label].append(y_val)
            if root_label in v_e_legend_map: # Добавляем v_e только если создали список
                 v_e_legend_map[root_label].append(v_e)

    # Дополняем списки до максимальной длины
    max_len = len(Ts)
    for data_dict in [all_vals, y_quantity_vals]:
        for root_label in data_dict.keys():
            if len(data_dict[root_label]) < max_len:
                data_dict[root_label].extend([np.nan] * (max_len - len(data_dict[root_label])))

    # --- Построение графика Stability vs T --- (код без изменений)
    plt.figure()
    for root_label, re_max_values in all_vals.items():
        # Рассчитываем среднее v_e для легенды, исключая nan
        valid_v_es = [v for v in v_e_legend_map.get(root_label, []) if not np.isnan(v)]
        mean_v_e_for_branch = np.mean(valid_v_es) if valid_v_es else np.nan
        label_text = f"{root_label} (avg v_e≈{mean_v_e_for_branch:.2f} m/s)"
        if np.isnan(mean_v_e_for_branch):
            label_text = root_label # Если нет v_e, просто имя ветви

        plt.plot(Ts, re_max_values, '-o', markersize=3, label=label_text)
    
    plt.axhline(0, linestyle='--', color='k')
    plt.title(f'Stability vs T ($\tau={tau}$ s, q={q*3600:.0f} veh/h)')
    plt.xlabel('T [s]'); plt.ylabel('max Re λ')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=min(val for lst in all_vals.values() for val in lst if not np.isnan(val))-0.1 if any(any(not np.isnan(v) for v in lst) for lst in all_vals.values()) else -1,
             top=max(val for lst in all_vals.values() for val in lst if not np.isnan(val)) + 0.1 if any(any(not np.isnan(v) for v in lst) for lst in all_vals.values()) else 1)
    plt.savefig(f'{figdir}/stability_vs_T_all_branches.png')
    plt.close()

    # --- Построение НОВОГО графика Y(T) vs T ---
    plt.figure()
    has_valid_y_data = False
    min_y_plot = float('inf')
    max_y_plot = float('-inf')

    for root_label, y_values in y_quantity_vals.items():
        # Получаем текст метки из v_e_legend_map
        valid_v_es = [v for v in v_e_legend_map.get(root_label, []) if not np.isnan(v)]
        mean_v_e_for_branch = np.mean(valid_v_es) if valid_v_es else np.nan
        label_text = f"{root_label} (avg v_e≈{mean_v_e_for_branch:.2f} m/s)"
        if np.isnan(mean_v_e_for_branch):
            label_text = root_label
        
        # Проверяем, есть ли хоть одно не-NaN значение для построения
        valid_y_points = [y for y in y_values if not np.isnan(y)]
        if valid_y_points:
            has_valid_y_data = True
            current_min = np.min(valid_y_points)
            current_max = np.max(valid_y_points)
            if current_min < min_y_plot: min_y_plot = current_min
            if current_max > max_y_plot: max_y_plot = current_max
            plt.plot(Ts, y_values, '-o', markersize=3, label=label_text)
    
    # Добавляем линию tau^2
    tau_squared = tau**2
    plt.axhline(tau_squared, color='r', linestyle='--', label=f'$\tau^2 = {tau_squared:.2f}$')
    
    plt.title(f'Y(T) = 1 / (0.5A + B - C) vs T ($\tau={tau}$ s, q={q*3600:.0f} veh/h)')
    plt.xlabel('T [s]')
    plt.ylabel(r'$1 / (\frac{1}{2}A + B - C)$') # Метка оси Y с LaTeX
    plt.legend()
    plt.grid(True)
    if has_valid_y_data:
        # Добавим небольшой запас к пределам
        margin = (max_y_plot - min_y_plot) * 0.1 if max_y_plot > min_y_plot else 0.1
        plt.ylim(bottom=min_y_plot - margin, top=max_y_plot + margin)
    else:
        plt.ylim(bottom=tau_squared - 1, top=tau_squared + 1) # Центрируем вокруг tau^2, если нет данных
    plt.savefig(f'{figdir}/Y_quantity_vs_T.png')
    plt.close()

# ==== 7. Main ====
def main():
    PARAMS = dict(
        a=2.6,        # m/s²
        b=4.5,        # comfortable decel
        v0=25.0,      # m/s
        delta=4,
        s0=5.0,       # m
        T=1,        # s (будет перезаписан в scan)
        L=1.0         # m
    )
    q_h = 900      # veh/h  — обязательно в veh/h!
    tau = 0.5        # s
    q   = q_h / 3600.0
    figdir = 'figs'
    os.makedirs(figdir, exist_ok=True)

    print(f'Parameters: {json.dumps(PARAMS)}  q_h={q_h}  tau={tau}')
    roots = equilibrium(PARAMS['v0'], PARAMS['delta'],
                        PARAMS['s0'], PARAMS['T'],
                        PARAMS['L'], q)
    if not roots:
        print('No equilibrium found for given q')
        return

    for i, v in enumerate(roots, start=1):
        print(f'Root{i}: v_e={v:.3f}')
        analyse_root(i, v, PARAMS.copy(), q, tau, figdir)

    scan_T(PARAMS.copy(), q, tau, figdir)
    print(f'All plots saved to ./{figdir}/')

if __name__ == '__main__':
    main()
