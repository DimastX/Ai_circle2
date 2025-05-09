import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq # Для поиска корней

# Стандартные параметры IDM из статьи (Таблица 1, s1=0)
DEFAULT_IDM_PARAMS = {
    'v0_desired_speed': 33.3,  # m/s (120 km/h)
    'T_safe_time_headway': 1.6, # s
    'a_max_accel': 0.73,       # m/s^2 (в статье это 'a')
    'b_comfort_decel': 1.67,   # m/s^2 (в статье это 'b')
    'delta_accel_exponent': 4, # безразмерный (в статье это 'delta' или 'd')
    's0_jam_distance': 2.0,    # m (чистая дистанция в заторе)
    'l_vehicle_length': 5.0,   # m (длина автомобиля)
    's1_gap_param': 0.0        # m (параметр s1, обычно 0 для IDM по статье)
}

def calculate_s_hat(v, dv, params):
    """
    Вычисляет желаемую динамическую дистанцию ŝ(v, Δv) (Ур. 14 в статье).
    Предполагается, что params['s1_gap_param'] = 0, как указано в статье для стандартных параметров IDM.
    """
    if params['s1_gap_param'] != 0:
        print("Предупреждение: Эта реализация оптимизирована для s1_gap_param = 0.")

    sqrt_term_val = params['a_max_accel'] * params['b_comfort_decel']
    if sqrt_term_val <= 0: 
        s_hat_val = params['s0_jam_distance'] + params['T_safe_time_headway'] * v
        if dv != 0:
            print(f"Предупреждение: Невозможно вычислить член с dv в s_hat из-за a*b <=0. ({sqrt_term_val})")
    else: 
        s_hat_val = params['s0_jam_distance'] + \
                params['T_safe_time_headway'] * v - \
                    (v * dv) / (2 * math.sqrt(sqrt_term_val))
    return s_hat_val

def calculate_idm_acceleration(s, dv, v, params):
    """
    Вычисляет ускорение согласно модели IDM f(s, Δv, v) (Ур. 13 в статье).
    s: текущая дистанция (от бампера до бампера)
    dv: относительная скорость (v_leader - v_follower)
    v: текущая скорость автомобиля
    params: словарь параметров IDM
    """
    if s <= params['l_vehicle_length']: 
        net_spacing = s - params['l_vehicle_length']
        if net_spacing <= 1e-3: 
            return -params['b_comfort_decel'] * 10 

    if params['v0_desired_speed'] <= 0:
        term_vel = 1.0 
    else:
        term_vel = (max(0,v) / params['v0_desired_speed'])**params['delta_accel_exponent']

    # Используем s_hat_calc для ясности, что это результат вызова функции
    s_hat_calc = calculate_s_hat(v, dv, params)
    
    net_clearance = s - params['l_vehicle_length']
    if net_clearance <= 1e-3: 
        term_spacing = (max(0, params['s0_jam_distance']) / (1e-3 if net_clearance <=0 else net_clearance) )**2 
    else:
        term_spacing = (s_hat_calc / net_clearance)**2
        
    acceleration = params['a_max_accel'] * (1 - term_vel - term_spacing)
    return acceleration

def find_equilibrium_velocity(s_star, params, tol=1e-6, max_iter=100):
    """
    Находит равновесную скорость v* для заданной равновесной дистанции s*
    путем решения f(s*, 0, v*) = 0 методом бисекции.
    """
    if s_star <= params['l_vehicle_length'] + params['s0_jam_distance']:
        if s_star < params['l_vehicle_length']: 
            return float('nan') 
        return 0.0

    v0_target = params['v0_desired_speed']
    if v0_target <= 0: 
        return 0.0

    def h_v_star(v_cand):
        if v_cand < 0: return float('inf') 
        net_clearance = s_star - params['l_vehicle_length']
        term1 = (max(0, v_cand) / v0_target)**params['delta_accel_exponent']
        s_hat_at_eq = params['s0_jam_distance'] + params['T_safe_time_headway'] * v_cand
        if net_clearance <= 1e-9: 
             return float('inf') if s_hat_at_eq > 0 else (1.0 if s_hat_at_eq == 0 else -1.0)
        term2 = (s_hat_at_eq / net_clearance)**2
        return term1 + term2 - 1

    low_v = 0.0
    high_v = v0_target * 1.1 
    h_low = h_v_star(low_v)
    if abs(h_low) < tol:
        return low_v
    h_high = h_v_star(high_v)
    if abs(h_high) < tol and h_high == 0 : 
         return high_v
    if h_low > 0 :
        return 0.0
    if h_low * h_high > 0:
        if h_high < 0: 
             if abs(h_v_star(params['v0_desired_speed'])) < tol : return params['v0_desired_speed']
        return float('nan') 

    for i in range(max_iter):
        mid_v = (low_v + high_v) / 2
        if mid_v == low_v or mid_v == high_v: 
            break
        h_mid = h_v_star(mid_v)
        if abs(h_mid) < tol or (high_v - low_v) / 2 < tol:
            return mid_v
        if h_low * h_mid < 0:
            high_v = mid_v
        else:
            low_v = mid_v
            h_low = h_mid 
    return mid_v

def calculate_s_star_for_fixed_v_star(fixed_v_star, params, tol=1e-9):
    """
    Вычисляет равновесную дистанцию s* для заданной равновесной скорости v*.
    """
    v0 = params['v0_desired_speed']
    l_veh = params['l_vehicle_length']
    s0 = params['s0_jam_distance']
    T = params['T_safe_time_headway']
    delta_exp = params['delta_accel_exponent']

    if fixed_v_star < 0 or v0 <= 0:
        return float('nan')
    if abs(fixed_v_star) < tol: 
        return l_veh + s0
    v_ratio = fixed_v_star / v0
    try:
        term_v_pow_delta = v_ratio ** delta_exp
    except ValueError: 
        return float('nan')
    C1 = 1.0 - term_v_pow_delta
    s_hat_at_fixed_v = s0 + T * fixed_v_star
    if abs(C1) < tol:  
        if abs(s_hat_at_fixed_v) < tol: 
            return float('inf')  
        else:
            return float('nan') 
    if C1 < 0: 
        return float('nan')
    try:
        sqrt_C1 = math.sqrt(C1)
        if abs(sqrt_C1) < tol: 
            return float('inf') if abs(s_hat_at_fixed_v) > tol else float('nan') 
        calculated_s_star = l_veh + abs(s_hat_at_fixed_v) / sqrt_C1
        if calculated_s_star < (l_veh + s0 - tol) and abs(fixed_v_star)>tol :
            return float('nan') 
        return calculated_s_star
    except (ValueError, ZeroDivisionError):
        return float('nan')

def find_equilibrium_state_for_flow(target_Q_veh_per_sec, params, 
                                  v_search_min_abs=1e-3, 
                                  xtol_brentq=1e-6, maxiter_brentq=100, 
                                  verbose=False):
    """
    Находит равновесную скорость v_e и дистанцию s_e для заданного потока Q.
    target_Q_veh_per_sec: поток в авто/сек.
    """
    l_veh = params['l_vehicle_length']
    s0 = params['s0_jam_distance']
    v0_desired = params['v0_desired_speed']

    if target_Q_veh_per_sec < 0:
        if verbose: print("Ошибка: Целевой поток Q не может быть отрицательным.")
        return None, None
    if abs(target_Q_veh_per_sec) < 1e-9: # Практически нулевой поток
        return 0.0, l_veh + s0

    # Целевая функция: G(v) = IDM_acceleration(s=v/Q, dv=0, v, params)
    # Мы ищем корень G(v) = 0.
    def objective_func_for_flow(v_cand):
        if v_cand < 0: # Скорость не может быть отрицательной
            return 1e12 # Большое значение, чтобы оттолкнуть решатель
        
        s_cand = v_cand / target_Q_veh_per_sec
        
        # Физические ограничения для s_cand
        if s_cand < l_veh + 1e-6: # s_cand должна быть хотя бы немного больше длины машины
            return 1e12 
        # Если v_cand > epsilon, то s_cand - l_veh должно быть >= s0
        if v_cand > 1e-3 and (s_cand - l_veh) < (s0 - 1e-6):
             return 1e12 # Нефизично: положительная скорость при зазоре меньше s0
        
        return calculate_idm_acceleration(s_cand, 0, v_cand, params)

    # Определение границ для поиска v_e
    # Нижняя граница: v должна быть такой, чтобы s = v/Q > l (лучше s > l + s0 для v > 0)
    v_lower_bound = target_Q_veh_per_sec * (l_veh + s0 + 1e-3) # Дает s = l+s0+eps_s
    # Однако, если Q очень велико, v_lower_bound может превысить v0_desired
    # Также, если Q очень мало, v_lower_bound может быть почти 0.
    # Минимальная абсолютная скорость для поиска (кроме v=0)
    v_min_for_search = max(v_search_min_abs, target_Q_veh_per_sec * (l_veh + 1e-3))
    
    v_upper_bound = v0_desired

    if v_min_for_search >= v_upper_bound - 1e-3: # Проверка если диапазон слишком мал или инвертирован
        if verbose: print(f"Диапазон поиска для v_e слишком мал или некорректен: [{v_min_for_search:.2f}, {v_upper_bound:.2f}] для Q={target_Q_veh_per_sec:.4f}")
        # Попробуем проверить, не является ли v=0 решением (для очень малых Q, где s->inf)
        if abs(objective_func_for_flow(v_search_min_abs)) < 1e-4 and v_search_min_abs < 0.1 : # если v_search_min_abs очень мал
             # Это может быть случай, когда Q очень мал, v стремится к v0, но objective(v_search_min_abs) оказывается близко к 0.
             # Лучше проверить на границах.
             pass # Продолжаем к brentq, он может не найти корень
        # Если target_Q настолько велик, что v_min_for_search > v0, то решения нет в [0,v0]
        # Это означает, что Q > Q_max (максимальной пропускной способности)
        return None, None
        
    try:
        # Проверяем знаки на концах интервала
        val_low = objective_func_for_flow(v_min_for_search)
        val_high = objective_func_for_flow(v_upper_bound)

        if verbose: print(f"Поиск v_e для Q={target_Q_veh_per_sec:.4f} в [{v_min_for_search:.3f}, {v_upper_bound:.3f}]. G(low)={val_low:.3f}, G(high)={val_high:.3f}")

        if val_low * val_high > 0: # Корня нет в интервале или их четное число
            # Может быть, Q > Q_max или Q очень мал и v_e -> v0, где objective_func_for_flow(v0) ~ 0
            if abs(val_high) < 1e-4 : # Если на v0 функция почти ноль
                v_e = v_upper_bound
                s_e = v_e / target_Q_veh_per_sec
                if verbose: print(f"Найдено решение на верхней границе: v_e={v_e:.2f}, s_e={s_e:.2f}")
                return v_e, s_e
            if abs(val_low) < 1e-4 and v_min_for_search < 0.1: # Если на нижней (очень малой v) функция почти ноль
                v_e = v_min_for_search # Это может быть v=0 в численном виде
                s_e = l_veh + s0 # Для v~0, s~l+s0
                if verbose: print(f"Найдено решение на нижней границе (v~0): v_e={v_e:.2f}, s_e={s_e:.2f}")
                return v_e, s_e
            if verbose: print(f"Не удалось найти интервал со сменой знака для brentq.")
            return None, None
        
        v_e = brentq(objective_func_for_flow, v_min_for_search, v_upper_bound, xtol=xtol_brentq, maxiter=maxiter_brentq)
        s_e = v_e / target_Q_veh_per_sec
        
        # Дополнительная проверка на физичность s_e
        if s_e < l_veh + s0 - 1e-3 and v_e > 1e-3:
             if verbose: print(f"Предупреждение: Найденное решение (v_e={v_e:.2f}, s_e={s_e:.2f}) для Q={target_Q_veh_per_sec:.4f} может быть нефизичным (s_e < l+s0 при v_e > 0).")
             # Это может указывать на то, что Q слишком велико или проблема с границами поиска
             return None, None
        if verbose: print(f"Найдено решение: v_e={v_e:.2f}, s_e={s_e:.2f}")
        return v_e, s_e
    except ValueError as e:
        # brentq может выбросить ValueError если f(a) и f(b) имеют одинаковый знак.
        if verbose: print(f"Ошибка brentq (ValueError): {e} при поиске v_e для Q={target_Q_veh_per_sec:.4f}")
        return None, None
    except RuntimeError as e:
        # brentq может выбросить RuntimeError если не сходится
        if verbose: print(f"Ошибка brentq (RuntimeError): {e} при поиске v_e для Q={target_Q_veh_per_sec:.4f}")
        return None, None

def calculate_partial_derivatives(s_star, v_star, params):
    """
    Вычисляет частные производные f_s, f_dv, f_v в точке равновесия (s*, 0, v*).
    Предполагается, что params['s1_gap_param'] = 0.
    """
    a_max = params['a_max_accel']
    l_veh = params['l_vehicle_length']
    s0 = params['s0_jam_distance']
    T = params['T_safe_time_headway']
    v0_target = params['v0_desired_speed']
    delta = params['delta_accel_exponent']
    b_decel = params['b_comfort_decel'] 

    if v0_target <= 0: 
        return 0.0, 0.0, 0.0 

    net_clearance = s_star - l_veh
    if net_clearance <= 1e-3:
        if v_star == 0 and abs(net_clearance - s0) < 1e-3 : 
             pass 
        elif v_star > 0 : 
             return float('nan'), float('nan'), float('nan')
        if net_clearance <= 1e-3: 
            return float('nan'), float('nan'), float('nan')

    s_hat_eq = s0 + T * v_star
    try:
        f_s = 2 * a_max * (s_hat_eq**2) / (net_clearance**3)
    except ZeroDivisionError:
        f_s = float('inf') if a_max * (s_hat_eq**2) > 0 else (float('-inf') if a_max * (s_hat_eq**2) < 0 else 0)

    sqrt_ab_val = a_max * b_decel
    if sqrt_ab_val <= 0 or v_star < 0: 
        f_dv = 0.0 
    else:
        denominator_f_dv = (net_clearance**2) * math.sqrt(sqrt_ab_val)
        if abs(denominator_f_dv) < 1e-9 : 
            f_dv = float('inf') if a_max * s_hat_eq * v_star > 0 else 0.0 
        else:
            f_dv = a_max * s_hat_eq * v_star / denominator_f_dv

    if v_star == 0:
        if delta == 1: term_v_ratio_deriv = -delta / v0_target
        elif delta < 1: term_v_ratio_deriv = -float('inf') 
        else: term_v_ratio_deriv = 0.0 
    elif v_star < 0 : 
        term_v_ratio_deriv = float('nan')
    else: 
        try:
            term_v_ratio_deriv = -delta / v0_target * (v_star / v0_target)**(delta - 1)
        except ZeroDivisionError: 
            term_v_ratio_deriv = float('nan')

    try:
        term_spacing_deriv = -2 * s_hat_eq / (net_clearance**2) * T 
    except ZeroDivisionError:
        term_spacing_deriv = float('-inf') if -2 * s_hat_eq * T > 0 else (float('inf') if -2 * s_hat_eq * T < 0 else 0)

    if math.isnan(term_v_ratio_deriv) or math.isnan(term_spacing_deriv):
        f_v = float('nan')
    elif math.isinf(term_v_ratio_deriv) or math.isinf(term_spacing_deriv):
        if math.isinf(term_v_ratio_deriv) and math.isinf(term_spacing_deriv) and \
           (term_v_ratio_deriv > 0) != (term_spacing_deriv > 0) : 
            f_v = float('nan')
        else:
            f_v = a_max * (term_v_ratio_deriv + term_spacing_deriv)
    else:
        f_v = a_max * (term_v_ratio_deriv + term_spacing_deriv)
    
    return f_s, f_dv, f_v

def check_rational_driving_constraints(f_s, f_dv, f_v):
    valid_fs = f_s > 1e-9 
    valid_fdv = f_dv >= -1e-9 
    valid_fv = f_v < -1e-9  
    return valid_fs and valid_fdv and valid_fv

def analyze_platoon_stability(f_s, f_dv, f_v, verbose=True):
    if verbose: print("--- Анализ устойчивости взвода (Platoon Stability) ---")
    if any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]):
        if verbose: print("Невозможно проанализировать: одна из производных NaN или Inf.")
        return False
    coeff_b = f_dv - f_v; coeff_c = f_s; is_stable = False
    if coeff_b > 1e-9 and coeff_c > 1e-9:
        is_stable = True
        if verbose: print(f"(f_Δv-f_v)={coeff_b:.4f}, f_s={coeff_c:.4f}. Раус-Гурвиц: УСТОЙЧИВ.")
    else:
        if verbose: print(f"(f_Δv-f_v)={coeff_b:.4f}, f_s={coeff_c:.4f}. Раус-Гурвиц не выполнен. Анализ корней:")
        try:
            D = coeff_b**2 - 4*coeff_c
            if D >= 0: 
                l1=(-coeff_b+math.sqrt(D))/2; l2=(-coeff_b-math.sqrt(D))/2;
                if verbose: print(f"Re(λ1)={l1:.4f}, Re(λ2)={l2:.4f}")
                is_stable = l1<-1e-9 and l2<-1e-9
            else:
                rp=-coeff_b/2; 
                if verbose: print(f"Re(λ)={rp:.4f}")
            is_stable = rp<-1e-9
            if is_stable and verbose: print("Результат: Взвод УСТОЙЧИВ (корни).")
            elif not is_stable and verbose: print("Результат: Взвод НЕУСТОЙЧИВ (корни).")
        except OverflowError: 
            is_stable=False; 
        if verbose: print("Overflow при анализе корней взвода.") 
    return is_stable

def analyze_string_stability(f_s, f_dv, f_v, verbose=True):
    if verbose: print("--- Анализ устойчивости цепочки (String Stability) ---")
    if any(math.isnan(x) or math.isinf(x) for x in [f_s,f_dv,f_v]):
        if verbose: print("NaN/Inf в производных.")
        return False
    if abs(f_v)<1e-9: 
        if verbose: print("f_v~0.")
        return False
    try:
        K = (f_v**2)/2 - f_dv*f_v - f_s
        if verbose: print(f"K = f_v^2/2 - f_Δv*f_v - f_s: {K:.4f}")
        is_stable = False
        if f_s>1e-9 and f_v<-1e-9: 
            is_stable = K>1e-9
        else:
            if verbose: print("Рациональное вождение нарушено.")
            is_stable = False 
        if verbose: print(f"Результат: Цепочка {'УСТОЙЧИВА' if is_stable else 'НЕУСТОЙЧИВА'}.")
    except (ZeroDivisionError, OverflowError): 
        is_stable=False; 
        if verbose: print("Ошибка вычисления K.")
    return is_stable

def collect_data_for_plots(s_star_range, params):
    s_values, v_values, fs_values, fdv_values, fv_values, k_values, platoon_flags, string_flags = [],[],[],[],[],[],[],[]
    for s_star in s_star_range:
        v_star = find_equilibrium_velocity(s_star, params)
        if v_star is None or math.isnan(v_star): continue
        f_s, f_dv, f_v = calculate_partial_derivatives(s_star, v_star, params)
        if any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]): continue
        s_values.append(s_star); v_values.append(v_star); fs_values.append(f_s); fdv_values.append(f_dv); fv_values.append(f_v)
        k_values.append((f_v**2)/2 - f_dv*f_v - f_s)
        rational = check_rational_driving_constraints(f_s, f_dv, f_v) 
        platoon_flags.append(analyze_platoon_stability(f_s, f_dv, f_v, verbose=False) if rational else False)
        string_flags.append(analyze_string_stability(f_s, f_dv, f_v, verbose=False) if rational else False)
    return {'s_star':np.array(s_values), 'v_star':np.array(v_values), 'f_s':np.array(fs_values), 'f_dv':np.array(fdv_values), 'f_v':np.array(fv_values), 'K_condition':np.array(k_values), 'platoon_stable':np.array(platoon_flags), 'string_stable':np.array(string_flags)}

def plot_stability_analysis(data, params):
    if len(data['s_star']) == 0: print("Нет данных для построения графиков s* vs v*."); return
    v_star_kmh = data['v_star'] * 3.6; plt.figure(figsize=(12, 8))
    plt.subplot(2,2,1); plt.plot(data['s_star'], v_star_kmh, 'b.-'); plt.xlabel('s* (м)'); plt.ylabel('v* (км/ч)'); plt.title('v* от s*'); plt.grid(True)
    ax_k = plt.subplot(2,2,2); idx=np.argsort(data['v_star']); plt.plot(v_star_kmh[idx], data['K_condition'][idx], 'r.-'); plt.axhline(0, color='k', lw=0.8, ls='--'); plt.xlabel('v* (км/ч)'); plt.ylabel('K'); plt.title('K от v*'); plt.grid(True)
    plt.subplot(2,2,3); 
    plt.plot(v_star_kmh[idx], data['f_s'][idx], marker='.', linestyle='-', label='f_s'); 
    plt.plot(v_star_kmh[idx], data['f_dv'][idx], marker='.', linestyle='-', label='f_Δv'); 
    plt.plot(v_star_kmh[idx], data['f_v'][idx], marker='.', linestyle='-', label='f_v'); 
    plt.xlabel('v* (км/ч)'); plt.ylabel('Производные'); plt.title('Производные от v*'); plt.legend(); plt.grid(True)
    ax_stab = plt.subplot(2,2,4)
    ps_mask = data['platoon_stable'][idx]; ss_mask = data['string_stable'][idx]
    valid_v_star_kmh = v_star_kmh[idx]
    if np.any(ps_mask): ax_stab.scatter(valid_v_star_kmh[ps_mask], np.ones(np.sum(ps_mask))*1.0, c='c', marker='o', label='Взвод Уст.', alpha=0.7)
    if np.any(~ps_mask): ax_stab.scatter(valid_v_star_kmh[~ps_mask], np.ones(np.sum(~ps_mask))*1.0, c='m', marker='x', label='Взвод Неуст.', alpha=0.7)
    if np.any(ss_mask): ax_stab.scatter(valid_v_star_kmh[ss_mask], np.ones(np.sum(ss_mask))*0.5, c='g', marker='o', label='Цеп. Уст.', alpha=0.7)
    if np.any(~ss_mask): ax_stab.scatter(valid_v_star_kmh[~ss_mask], np.ones(np.sum(~ss_mask))*0.5, c='r', marker='x', label='Цеп. Неуст.', alpha=0.7)
    ax_stab.set_yticks([0.5,1.0]); ax_stab.set_yticklabels(['Цепочка','Взвод']); ax_stab.set_xlabel('v* (км/ч)'); ax_stab.set_title('Устойчивость'); ax_stab.set_ylim(0,1.5); ax_stab.legend(fontsize='small'); ax_stab.grid(True)
    plt.suptitle(f"Анализ IDM (s* vs v*)\nParams: a={params['a_max_accel']:.2f}, T={params['T_safe_time_headway']:.2f}, b={params['b_comfort_decel']:.2f}, s0={params['s0_jam_distance']:.2f}", fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.95]); plt.show()

def collect_data_for_param_sweep(
    param_to_sweep_key, sweep_values, base_idm_params, 
    fixed_s_star=None, fixed_v_star=None, fixed_Q=None, 
    verbose=False):
    
    param_vals_list, s_star_list, v_star_list, K_list, platoon_stable_list, string_stable_list = [],[],[],[],[],[]

    num_fixed_conditions = sum(x is not None for x in [fixed_s_star, fixed_v_star, fixed_Q])
    if num_fixed_conditions == 0:
        raise ValueError("Необходимо зафиксировать s_star, v_star или Q.")
    if num_fixed_conditions > 1:
        raise ValueError("Можно зафиксировать только одну переменную: s_star, v_star или Q.")

    for val in sweep_values:
        current_params = base_idm_params.copy()
        current_params[param_to_sweep_key] = val
        current_s, current_v = None, None

        if fixed_s_star is not None:
            current_s = fixed_s_star
            current_v = find_equilibrium_velocity(current_s, current_params)
        elif fixed_v_star is not None:
            current_v = fixed_v_star
            current_s = calculate_s_star_for_fixed_v_star(current_v, current_params)
        elif fixed_Q is not None:
            current_v, current_s = find_equilibrium_state_for_flow(fixed_Q, current_params, verbose=verbose)
        
        if current_s is None or math.isnan(current_s) or math.isinf(current_s) or \
           current_v is None or math.isnan(current_v) or math.isinf(current_v):
            if verbose: print(f"Пропуск {param_to_sweep_key}={val}: s* или v* невалидны (s*={current_s}, v*={current_v})")
            continue
        
        f_s, f_dv, f_v = calculate_partial_derivatives(current_s, current_v, current_params)
        if any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]):
            if verbose: print(f"Пропуск {param_to_sweep_key}={val}: NaN/Inf в производных (s*={current_s:.2f}, v*={current_v:.2f})")
            continue
        
        param_vals_list.append(val); s_star_list.append(current_s); v_star_list.append(current_v)
        K_list.append((f_v**2)/2 - f_dv*f_v - f_s)
        rational = check_rational_driving_constraints(f_s, f_dv, f_v)
        platoon_stable_list.append(analyze_platoon_stability(f_s, f_dv, f_v, verbose=False) if rational else False)
        string_stable_list.append(analyze_string_stability(f_s, f_dv, f_v, verbose=False) if rational else False)
        
    return {
        'param_values': np.array(param_vals_list), 's_star': np.array(s_star_list),
        'v_star': np.array(v_star_list), 'K_condition': np.array(K_list),
        'platoon_stable': np.array(platoon_stable_list), 'string_stable': np.array(string_stable_list)
    }

def plot_stability_for_parameter_sweep(data, swept_param_key, swept_param_label, fixed_condition_label, base_params):
    if len(data['param_values']) == 0:
        print(f"Нет данных для построения графиков для параметра {swept_param_label}.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    param_x_values = data['param_values']

    if "Q =" in fixed_condition_label:
        axs0_twin = axs[0].twinx()
        line1, = axs[0].plot(param_x_values, data['v_star'] * 3.6, marker='.', linestyle='-', color='blue', alpha=0.8, label='v* (км/ч)')
        line2, = axs0_twin.plot(param_x_values, data['s_star'], linestyle='-', color='green', alpha=0.8, label='s* (м)')
        
        axs[0].set_ylabel('Равновесная скорость v* (км/ч)', color=line1.get_color())
        axs0_twin.set_ylabel('Равновесная дистанция s* (м)', color=line2.get_color())
        axs[0].tick_params(axis='y', labelcolor=line1.get_color())
        axs0_twin.tick_params(axis='y', labelcolor=line2.get_color())
        
        lines = [line1, line2]
        axs[0].legend(lines, [l.get_label() for l in lines], loc='best')
        axs[0].set_title(f'Зависимость v* и s* от {swept_param_label}')
    elif fixed_condition_label.startswith("s*"):
        y_var = data['v_star'] * 3.6 
        y_label = 'Равновесная скорость v* (км/ч)'
        axs[0].plot(param_x_values, y_var, marker='.', linestyle='-', color='b')
        axs[0].set_ylabel(y_label)
        axs[0].set_title(f'{y_label.split(" (")[0]} vs. {swept_param_label}')
    else:
        y_var = data['s_star']
        y_label = 'Равновесная дистанция s* (м)'
        axs[0].plot(param_x_values, y_var, marker='.', linestyle='-', color='b')
        axs[0].set_ylabel(y_label)
        axs[0].set_title(f'{y_label.split(" (")[0]} vs. {swept_param_label}')
    axs[0].grid(True)

    axs[1].plot(param_x_values, data['K_condition'], marker='.', linestyle='-', color='r')
    axs[1].axhline(0, color='black', lw=0.8, linestyle='--'); axs[1].set_ylabel('Критерий K')
    axs[1].set_title(f'Критерий устойчивости цепочки K vs. {swept_param_label}'); axs[1].grid(True)

    ps_mask = data['platoon_stable']; ss_mask = data['string_stable']
    valid_indices = ~np.isnan(param_x_values) & ~np.isnan(data['K_condition']) 
    
    px_valid = param_x_values[valid_indices]
    ps_mask_valid = ps_mask[valid_indices]
    ss_mask_valid = ss_mask[valid_indices]

    if np.any(ps_mask_valid): axs[2].scatter(px_valid[ps_mask_valid], np.ones(np.sum(ps_mask_valid))*1.0, c='c', marker='o', label='Взвод Уст.', alpha=0.7)
    if np.any(~ps_mask_valid): axs[2].scatter(px_valid[~ps_mask_valid], np.ones(np.sum(~ps_mask_valid))*1.0, c='m', marker='x', label='Взвод Неуст.', alpha=0.7)
    if np.any(ss_mask_valid): axs[2].scatter(px_valid[ss_mask_valid], np.ones(np.sum(ss_mask_valid))*0.5, c='g', marker='o', label='Цеп. Уст.', alpha=0.7)
    if np.any(~ss_mask_valid): axs[2].scatter(px_valid[~ss_mask_valid], np.ones(np.sum(~ss_mask_valid))*0.5, c='r', marker='x', label='Цеп. Неуст.', alpha=0.7)
    
    axs[2].set_yticks([0.5,1.0]); axs[2].set_yticklabels(['Цепочка','Взвод']); axs[2].set_xlabel(swept_param_label)
    axs[2].set_title('Области устойчивости'); axs[2].set_ylim(0,1.5); axs[2].legend(fontsize='small'); axs[2].grid(True)

    param_details_list = []
    for k_param, v_param in DEFAULT_IDM_PARAMS.items():
        if k_param == swept_param_key: continue
        current_val = base_params.get(k_param, v_param)
        param_details_list.append(f"{k_param.split('_')[0]}={current_val:.2f}")
    param_details = ", ".join(param_details_list)
    
    fig.suptitle(f'Анализ уст-ти IDM: {swept_param_label} ({fixed_condition_label})\nОст. параметры: {param_details}', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.show()

if __name__ == '__main__':
    params = DEFAULT_IDM_PARAMS.copy()
    
    s_star_example = 30.0
    print(f"--- Анализ для ОДНОЙ точки: s* = {s_star_example} м ---")
    v_star = find_equilibrium_velocity(s_star_example, params)
    if v_star is not None and not math.isnan(v_star):
        print(f"Равновесная скорость v* = {v_star:.2f} м/с ({(v_star*3.6):.2f} км/ч)")
        f_s,f_dv,f_v = calculate_partial_derivatives(s_star_example,v_star,params)
        if not any(math.isnan(x) or math.isinf(x) for x in [f_s,f_dv,f_v]):
            if check_rational_driving_constraints(f_s,f_dv,f_v):
                analyze_platoon_stability(f_s,f_dv,f_v, verbose=True)
                analyze_string_stability(f_s,f_dv,f_v, verbose=True)
            else:
                print("Условия рац. вождения не выполнены, анализ уст-ти не проводится.")
        else: print("Производные не определены.")
    else: print("Равновесная скорость не найдена.")
    print("-"*50)

    print("\n--- Графики: Зависимость от s* (Фундаментальная диаграмма и K) ---")
    min_s_star = params['l_vehicle_length'] + params['s0_jam_distance'] + 0.1
    max_s_star = 150 
    s_star_values_for_plot = np.linspace(min_s_star, max_s_star, 100)
    plot_data_s_v = collect_data_for_plots(s_star_values_for_plot, params.copy())
    if plot_data_s_v and len(plot_data_s_v['s_star']) > 0 :
        plot_stability_analysis(plot_data_s_v, params.copy())
    print("-"*50)

    print("\n--- Графики: Варьирование T (время реакции) при фикс. s* ---")
    fixed_s_for_T_sweep = 30.0
    T_sweep_values = np.linspace(0.5, 3.0, 50)
    data_sweep_T_fixed_s = collect_data_for_param_sweep(
        param_to_sweep_key='T_safe_time_headway', sweep_values=T_sweep_values,
        base_idm_params=params.copy(), fixed_s_star=fixed_s_for_T_sweep, verbose=False
    )
    if data_sweep_T_fixed_s and len(data_sweep_T_fixed_s['param_values']) > 0:
        plot_stability_for_parameter_sweep(
            data_sweep_T_fixed_s, swept_param_key='T_safe_time_headway',
            swept_param_label='Время реакции T (с)', 
            fixed_condition_label=f"s* = {fixed_s_for_T_sweep:.1f} м",
            base_params=params.copy()
        )
    print("-"*50)

    print("\n--- Графики: Варьирование 'a' (макс. ускорение) при фикс. v* ---")
    fixed_v_for_a_sweep = 20.0  
    a_sweep_values = np.linspace(0.3, 2.0, 50) 
    data_sweep_a_fixed_v = collect_data_for_param_sweep(
        param_to_sweep_key='a_max_accel', sweep_values=a_sweep_values,
        base_idm_params=params.copy(), fixed_v_star=fixed_v_for_a_sweep, verbose=False
    )
    if data_sweep_a_fixed_v and len(data_sweep_a_fixed_v['param_values']) > 0:
        plot_stability_for_parameter_sweep(
            data_sweep_a_fixed_v, swept_param_key='a_max_accel',
            swept_param_label='Макс. ускорение a (м/с²)', 
            fixed_condition_label=f"v* = {fixed_v_for_a_sweep * 3.6:.1f} км/ч ({fixed_v_for_a_sweep:.1f} м/с)",
            base_params=params.copy()
        )
    print("-"*50)

    print("\n--- Графики: Варьирование T (время реакции) при фикс. Q ---")
    target_Q_veh_per_hour = 1800 
    target_Q_veh_per_sec = target_Q_veh_per_hour / 3600.0 
    T_sweep_values_for_Q = np.linspace(0.8, 2.5, 50) 
    
    data_sweep_T_fixed_Q = collect_data_for_param_sweep(
        param_to_sweep_key='T_safe_time_headway', 
        sweep_values=T_sweep_values_for_Q,
        base_idm_params=params.copy(),
        fixed_Q=target_Q_veh_per_sec,
        verbose=False 
    )
    if data_sweep_T_fixed_Q and len(data_sweep_T_fixed_Q['param_values']) > 0:
        plot_stability_for_parameter_sweep(
            data_sweep_T_fixed_Q, 
            swept_param_key='T_safe_time_headway',
            swept_param_label='Время реакции T (с)', 
            fixed_condition_label=f"Q = {target_Q_veh_per_hour:.0f} авто/час",
            base_params=params.copy()
        )
    else:
        print(f"Не удалось собрать данные для варьирования T при Q={target_Q_veh_per_hour:.0f} авто/час.")

    print("\nАнализ завершен.") 