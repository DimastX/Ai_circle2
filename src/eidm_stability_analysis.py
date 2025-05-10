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
    Вычисляет желаемую динамическую дистанцию ŝ(v, Δv) (Ур. 14 в статье) - это желаемый чистый зазор.
    Предполагается, что params['s1_gap_param'] = 0, как указано в статье для стандартных параметров IDM.
    """
    if params['s1_gap_param'] != 0:
        print("Предупреждение: Эта реализация оптимизирована для s1_gap_param = 0.")

    sqrt_term_val = params['a_max_accel'] * params['b_comfort_decel']
    if sqrt_term_val <= 1e-9: # Избегаем деления на ноль или корень из очень малого числа
        s_hat_val = params['s0_jam_distance'] + params['T_safe_time_headway'] * v
        if dv != 0 and sqrt_term_val > 0: # Если все же есть очень малое значение
             s_hat_val -= (v * dv) / (2 * math.sqrt(sqrt_term_val))
        elif dv !=0 :
             print(f"Предупреждение: Невозможно вычислить член с dv в s_hat из-за a*b <=0. ({sqrt_term_val})")
    else: 
        s_hat_val = params['s0_jam_distance'] + \
                params['T_safe_time_headway'] * v - \
                    (v * dv) / (2 * math.sqrt(sqrt_term_val))
    return max(0, s_hat_val) # Желаемый чистый зазор не может быть отрицательным

def calculate_idm_acceleration(s_net_clearance, dv, v, params):
    """
    Вычисляет ускорение согласно модели IDM f(s_net, Δv, v) (Ур. 13 в статье).
    s_net_clearance: текущий чистый зазор (от переднего бампера до заднего бампера переднего авто)
    dv: относительная скорость (v_leader - v_follower)
    v: текущая скорость автомобиля
    params: словарь параметров IDM
    """
    s0 = params['s0_jam_distance']

    if params['v0_desired_speed'] <= 0:
        term_vel = 1.0 
    else:
        term_vel = (max(0,v) / params['v0_desired_speed'])**params['delta_accel_exponent']

    s_hat_calc = calculate_s_hat(v, dv, params) 
    
    # Обработка s_net_clearance для term_spacing
    if s_net_clearance < 1e-3: # Если чистый зазор очень мал или отрицателен
        if s_net_clearance < 0: # столкновение
            term_spacing = (s_hat_calc / 1e-3)**2 * 100 # Очень большое значение для сильного торможения
        elif s_hat_calc <= 0: # s_hat=0 (не должно быть если s0>0)
            term_spacing = 0 # если желаемый зазор 0, то и компонента зазора 0 (если s_net >0)
        else: # s_net_clearance очень мал, но положителен
            term_spacing = (s_hat_calc / s_net_clearance)**2
    else: # Нормальный случай s_net_clearance >= s0
        term_spacing = (s_hat_calc / s_net_clearance)**2
        
    acceleration = params['a_max_accel'] * (1 - term_vel - term_spacing)
    return acceleration

def find_equilibrium_velocity(s_star_net, params, tol=1e-6, max_iter=100):
    """
    Находит равновесную скорость v* для заданной равновесной чистой дистанции s_star_net
    путем решения f(s_star_net, 0, v*) = 0.
    s_star_net: равновесный чистый зазор.
    """
    s0 = params['s0_jam_distance']
    v0_target = params['v0_desired_speed']

    if s_star_net < 0: 
        return float('nan')
    
    if v0_target <= 0: # Если желаемая скорость 0, то равновесная скорость 0 (s_star_net должен быть s0)
        if abs(s_star_net - s0) < tol:
            return 0.0
        else: # Несоответствие: v0=0 требует s_net=s0
            return float('nan') 

    # Если s_star_net = s0, то v* = 0. Проверим это явно.
    accel_at_v0_s_star_net_is_s0 = calculate_idm_acceleration(s0, 0, 0.0, params)
    if abs(s_star_net - s0) < tol and abs(accel_at_v0_s_star_net_is_s0) < tol :
        return 0.0

    # Если s_star_net < s0, то IDM будет давать отрицательное ускорение для v > 0, значит v_eq=0.
    # Однако, это должно быть согласовано с s_star_net = s0 для v_eq=0.
    if s_star_net < s0 - tol:
        # Это нефизичная ситуация для равновесия с v > 0.
        # Если s_star_net < s0, единственное устойчивое равновесие v=0 при s_net=s0.
        # Поскольку дан s_star_net < s0, такого равновесия нет.
        return float('nan') 


    def h_v_star(v_cand):
        if v_cand < 0: return float('inf') 
        return calculate_idm_acceleration(s_star_net, 0, v_cand, params)

    low_v = 0.0
    high_v = v0_target 

    try:
        h_low = h_v_star(low_v) # Ускорение при v=0 и данном s_star_net
        if abs(h_low) < tol: # Если при v=0 ускорение 0
             # Это возможно если s_star_net = s0 (уже проверено) или если s_star_net -> inf и v0=0 (не здесь)
            if abs(s_star_net-s0) < tol : return low_v # Подтверждаем v=0 для s_star_net=s0
            # Если h_low=0, но s_star_net != s0, это может быть артефакт, если s_star_net очень большой
            # и v0_target = 0, и (v/v0)^d член дает 1. Но v0_target > 0 здесь.
            # Значит, если h_low=0, то это v=0 и s_star_net=s0.

        h_high = h_v_star(high_v) # Ускорение при v=v0_target и данном s_star_net
        if abs(h_high) < tol: # Если при v=v0_target ускорение 0
            return high_v

        if h_low * h_high > 0:
            # Если h_low > 0 (s_star_net > s0) и h_high > 0 (ускорение на v0), то v_eq > v0 (не ловим brentq) или нет решения.
            # Если h_low < 0 (s_star_net < s0, уже должно было отсеяться) и h_high < 0.
            # print(f"find_eq_v: No sign change. s_net={s_star_net:.2f}, h_low={h_low:.3f}, h_high={h_high:.3f}")
            if h_low < 0 : return 0.0 # Если начинается с торможения и на v0 торможение, то v_eq=0 (или уже отсеяно s_star_net < s0)
            return float('nan') 

        v_star_candidate = brentq(h_v_star, low_v, high_v, xtol=tol, maxiter=max_iter)
        return v_star_candidate

    except ValueError:
        # print(f"ValueError in brentq for s_star_net={s_star_net:.2f}")
        # Повторная проверка границ
        if abs(h_v_star(low_v)) < tol * 10 :
             if abs(s_star_net-s0) < tol*10: return 0.0
        if abs(h_v_star(high_v)) < tol * 10 : return high_v
        return float('nan')
    except RuntimeError:
        # print(f"RuntimeError in brentq for s_star_net={s_star_net:.2f}")
        return float('nan')

def calculate_s_star_for_fixed_v_star(fixed_v_star, params, tol=1e-9):
    """
    Вычисляет равновесный чистый зазор s_star_net для заданной равновесной скорости v*.
    Возвращает чистый зазор.
    """
    v0 = params['v0_desired_speed']
    s0 = params['s0_jam_distance']
    T = params['T_safe_time_headway']
    delta_exp = params['delta_accel_exponent']

    if fixed_v_star < -tol : return float('nan')
    if v0 < 0 : return float('nan') # v0 не может быть отрицательной

    if abs(fixed_v_star) < tol: 
        return s0 
    
    if abs(v0) < tol : # v0 = 0
        return float('nan') # Если v0=0, то для v_star > 0 нет равновесия (IDM даст -inf accel)

    if fixed_v_star > v0 + tol : 
        return float('nan') # Равновесие не может быть при скорости больше желаемой v0

    term_v_pow_delta = (fixed_v_star / v0)**delta_exp
    C1 = 1.0 - term_v_pow_delta 
    s_hat_at_fixed_v = calculate_s_hat(fixed_v_star, 0, params) # Используем функцию для s_hat

    if C1 < tol and C1 > -tol: # C1 ~ 0 (fixed_v_star ~ v0)
        return float('inf') if s_hat_at_fixed_v > tol else s0 # Если s_hat=0 (нереально), то s0, иначе inf
            
    if C1 <= 0: # fixed_v_star > v0 (уже отсеяно) или C1=0 (выше)
        return float('nan') 

    try:
        sqrt_C1 = math.sqrt(C1) 
        calculated_s_star_net = s_hat_at_fixed_v / sqrt_C1
        
        if calculated_s_star_net < s0 - tol :
            return float('nan') 
        return calculated_s_star_net
    except (ValueError, ZeroDivisionError):
        return float('nan')

def find_equilibrium_state_for_flow(target_Q_veh_per_sec, params, 
                                  v_search_min_abs=1e-3, 
                                  xtol_brentq=1e-6, maxiter_brentq=100, 
                                  verbose=False):
    """
    Находит равновесную скорость v_e и чистый зазор s_e_net для заданного потока Q.
    target_Q_veh_per_sec: поток в авто/сек.
    Возвращает (v_e, s_e_net)
    """
    l_veh = params['l_vehicle_length']
    s0 = params['s0_jam_distance']
    v0_desired = params['v0_desired_speed']

    if target_Q_veh_per_sec < -1e-9:
        if verbose: print("Ошибка: Целевой поток Q не может быть отрицательным.")
        return None, None
    
    if abs(target_Q_veh_per_sec) < 1e-9: 
        return 0.0, s0

    def objective_func_for_flow(v_cand):
        if v_cand < 0: return 1e12 
        
        s_total_cand = v_cand / target_Q_veh_per_sec
        s_net_cand = s_total_cand - l_veh
        
        if s_net_cand < 0 : # Физически v/Q не может быть < L
             return 1e12 
        
        if abs(v_cand) < 1e-3 : # Для v~0, ожидаем s_net_cand ~ s0
            accel = calculate_idm_acceleration(s0, 0, 0.0, params) # Ускорение должно быть 0
            # Мы хотим, чтобы s_net_cand, вычисленный из Q и v_cand, был s0.
            # Если s_net_cand (v_cand/Q - L) не равен s0, то это не корень.
            # Вернем разницу или большое значение, если s_net_cand != s0
            # Это сложная точка. Если v_cand=0, то Q=0. Если Q!=0, v_cand не может быть 0.
            # Brentq ищет accel=0. Для v_cand~0, s_net~s0, accel~0.
            # Эта ветка для v_cand~0, где s_net_cand может быть не s0.
            return calculate_idm_acceleration(s_net_cand, 0, v_cand, params)

        # Если v_cand > 0 и s_net_cand < s0 (существенно)
        if s_net_cand < s0 - 1e-3: # Даем небольшой допуск
             return 1e12 # Нефизично, штраф

        return calculate_idm_acceleration(s_net_cand, 0, v_cand, params)

    # Границы поиска v_e: v от 0 до v0_desired
    # Нижняя граница v_min_for_search должна обеспечивать s_net >= 0 (т.е. v/Q >= L => v >= QL)
    # И желательно s_net >= s0 (т.е. v/Q >= L+s0 => v >= Q(L+s0))
    
    v_lower_bound_for_s_net_positive = target_Q_veh_per_sec * l_veh
    # v_lower_bound_for_s_net_s0 = target_Q_veh_per_sec * (l_veh + s0)

    v_min_for_search = max(v_search_min_abs, v_lower_bound_for_s_net_positive + 1e-6) # Гарантируем s_net > 0
    v_min_for_search = max(0.0, v_min_for_search) # Неотрицательность
    
    v_upper_bound = v0_desired

    if v_min_for_search >= v_upper_bound - xtol_brentq: 
        if verbose: print(f"Диапазон поиска для v_e ({v_min_for_search:.2f} - {v_upper_bound:.2f}) слишком мал для Q={target_Q_veh_per_sec:.4f}")
        # Проверим v=0 (s_net=s0) если Q=0 (уже обработано)
        # Если Q > 0, но диапазон схлопнулся, возможно Q > Q_max
        # Попробуем проверить границы objective_func_for_flow(0) и objective_func_for_flow(v0_desired)
        obj_at_zero_v = objective_func_for_flow(1e-4) # Проверка около нуля
        if abs(obj_at_zero_v) < xtol_brentq*10 :
             s_net_check = 1e-4 / target_Q_veh_per_sec - l_veh
             if abs(s_net_check - s0) < 1e-2 : # Если это состояние v~0, s_net~s0
                if verbose: print(f"Решение Q={target_Q_veh_per_sec:.4f} найдено на v~0, s_net~s0")
                return 1e-4, s0 # Возвращаем малую скорость и s0
        return None, None
            
    try:
        val_low = objective_func_for_flow(v_min_for_search)
        val_high = objective_func_for_flow(v_upper_bound)

        if verbose: print(f"Поиск v_e для Q={target_Q_veh_per_sec:.4f} в [{v_min_for_search:.3f}, {v_upper_bound:.3f}]. G(low)={val_low:.3f}, G(high)={val_high:.3f}")

        if val_low * val_high > 0: 
            if abs(val_high) < xtol_brentq*10 : 
                s_e_total_cand = v_upper_bound / target_Q_veh_per_sec
                s_e_net_cand = s_e_total_cand - l_veh
                if s_e_net_cand >= s0 - 1e-2: 
                    if verbose: print(f"Найдено решение на v_e=v0: v_e={v_upper_bound:.2f}, s_net_e={s_e_net_cand:.2f}")
                    return v_upper_bound, s_e_net_cand
            if abs(val_low) < xtol_brentq*10 :
                s_e_total_cand = v_min_for_search / target_Q_veh_per_sec
                s_e_net_cand = s_e_total_cand - l_veh
                if s_e_net_cand >= (s0 if abs(v_min_for_search) < 1e-2 else -1e-2) - 1e-2 :
                    if verbose: print(f"Найдено решение на v_e=v_min_for_search: v_e={v_min_for_search:.2f}, s_net_e={s_e_net_cand:.2f}")
                    return v_min_for_search, s_e_net_cand

            if verbose: print(f"Не удалось найти интервал со сменой знака для brentq (Q={target_Q_veh_per_sec:.4f}).")
            return None, None
        
        v_e = brentq(objective_func_for_flow, v_min_for_search, v_upper_bound, xtol=xtol_brentq, maxiter=maxiter_brentq)
        s_e_total = v_e / target_Q_veh_per_sec
        s_e_net = s_e_total - l_veh
        
        # Финальные проверки физичности
        if abs(v_e) < 1e-3: # Если скорость близка к нулю
            if abs(s_e_net - s0) > 1e-2 : # А чистый зазор не s0
                 if verbose: print(f"Коррекция: Для v_e~0 ({v_e:.3f}), s_e_net ({s_e_net:.3f}) приведен к s0 ({s0}).")
                 s_e_net = s0 # Приводим к s0
        elif v_e > 1e-3 and s_e_net < s0 - 1e-3 : 
             if verbose: print(f"Предупреждение: Нефизичное решение (s_e_net < s0 при v>0) для Q={target_Q_veh_per_sec:.4f}: v_e={v_e:.2f}, s_e_net={s_e_net:.2f}.")
             return None, None # Отбрасываем
        
        if s_e_net < -1e-3 : # Чистый зазор не может быть отрицательным
             if verbose: print(f"Предупреждение: Отрицательный чистый зазор для Q={target_Q_veh_per_sec:.4f}: v_e={v_e:.2f}, s_e_net={s_e_net:.2f}.")
             return None, None

        if verbose: print(f"Найдено решение: v_e={v_e:.2f}, s_e_net={s_e_net:.2f}")
        return v_e, s_e_net
    except ValueError as e:
        if verbose: print(f"Ошибка brentq (ValueError): {e} при поиске v_e для Q={target_Q_veh_per_sec:.4f}")
        return None, None
    except RuntimeError as e:
        if verbose: print(f"Ошибка brentq (RuntimeError): {e} при поиске v_e для Q={target_Q_veh_per_sec:.4f}")
        return None, None

def find_all_equilibrium_states_for_flow(target_Q_veh_per_sec, params, 
                                         v_search_min_abs=1e-3, 
                                         xtol_brentq=1e-6, maxiter_brentq=100,
                                         num_scan_intervals=200,
                                         verbose=False):
    """
    Находит все равновесные скорости v_e и чистые зазоры s_e_net для заданного потока Q.
    Возвращает список кортежей [(v_e, s_e_net), ...].
    """
    l_veh = params['l_vehicle_length']
    s0 = params['s0_jam_distance']
    v0_desired = params['v0_desired_speed']
    results = []

    if target_Q_veh_per_sec < -1e-9:
        if verbose: print("Ошибка: Целевой поток Q не может быть отрицательным.")
        return []
    
    if abs(target_Q_veh_per_sec) < 1e-9: 
        # Проверим, что это действительно равновесие
        accel_at_zero_flow = calculate_idm_acceleration(s0, 0, 0.0, params)
        if abs(accel_at_zero_flow) < xtol_brentq * 10:
            return [(0.0, s0)]
        else:
            if verbose: print(f"Предупреждение: Для Q=0, состояние (v=0, s_net=s0) не является равновесным (accel={accel_at_zero_flow:.3f})")
            return []

    # --- Начало определения внутренней функции objective_func_for_flow ---
    def objective_func_for_flow(v_cand):
        if v_cand < 0: return 1e12 # Штраф за отрицательную скорость
        
        # s_total_cand вычисляется как v_cand / target_Q_veh_per_sec
        # Но если v_cand очень мало, а Q велико, s_total_cand может быть очень мало.
        # s_net_cand = s_total_cand - l_veh
        
        if abs(v_cand) < 1e-9: # Если скорость почти ноль
            if target_Q_veh_per_sec > 1e-9: # А поток не ноль
                # Это означает, что s_total стремится к нулю, s_net к -L. Нефизично.
                # IDM должен давать большое отрицательное ускорение.
                # Однако, для поиска корня accel=0, это не решение.
                # Мы можем просто вернуть большое значение, чтобы brentq не нашел здесь корень,
                # если только не Q=0 (уже обработано).
                return 1e12 # Большое значение, чтобы избежать этого как решения для Q > 0
            else: # v_cand ~ 0 и Q ~ 0, это случай Q=0 (см. выше)
                 # Здесь мы ищем корень accel(s_net_cand, 0, v_cand) = 0.
                 # Если Q=0, то s_net_cand не определен из Q.
                 # Этот путь не должен достигаться, если Q=0 обработан.
                 return calculate_idm_acceleration(s0, 0, 0.0, params)


        s_total_cand = v_cand / target_Q_veh_per_sec
        s_net_cand = s_total_cand - l_veh
        
        # Физические ограничения на s_net_cand
        if s_net_cand < 0 : # Чистый зазор не может быть отрицательным (v/Q < L)
             return 1e12  # Штраф

        # Если v_cand > 0 и s_net_cand < s0 (существенно)
        # Для равновесия с v_cand > 0, мы ожидаем s_net_cand >= s0.
        # Если s_net_cand < s0, IDM должен давать положительное ускорение (если не затор),
        # или если это состояние затора (v_cand -> 0), то s_net_cand -> s0.
        # Если мы ищем accel=0, и s_net_cand < s0 при v_cand > 0, это может быть неустойчивая ветвь.
        # Для brentq, если accel(s_net_cand_low_s0, 0, v_cand) = 0, это математический корень.
        # Мы отфильтруем нефизичные решения позже.
        # Однако, если s_net_cand слишком сильно меньше s0, IDM формула может дать проблемы.
        # Допустим, s_net_cand может быть немного меньше s0 если это корень.

        # Особый случай: если s_net_cand очень близок к s0, а v_cand тоже очень близок к 0.
        # Это по сути состояние (v=0, s_net=s0).
        if abs(v_cand) < xtol_brentq and abs(s_net_cand - s0) < xtol_brentq * 10:
            # Вернем ускорение для (s0, 0, 0)
            return calculate_idm_acceleration(s0, 0, 0.0, params)
            
        return calculate_idm_acceleration(s_net_cand, 0, v_cand, params)
    # --- Конец определения внутренней функции objective_func_for_flow ---

    v_min_scan_boundary = target_Q_veh_per_sec * l_veh + 1e-6 # Гарантируем s_net > 0
    v_min_scan_boundary = max(v_search_min_abs, v_min_scan_boundary) # Неотрицательность
    v_min_scan_boundary = max(0.0, v_min_scan_boundary)
    
    v_max_scan_boundary = v0_desired

    if v_min_scan_boundary >= v_max_scan_boundary - xtol_brentq:
        if verbose: print(f"Диапазон сканирования для v_e ({v_min_scan_boundary:.3f} - {v_max_scan_boundary:.3f}) слишком мал для Q={target_Q_veh_per_sec:.4f}")
        # Попробуем проверить единственную точку, если диапазон схлопнулся
        # Это может быть максимальный поток.
        # Проверим, является ли v_max_scan_boundary (или v_min_scan_boundary) решением.
        obj_at_limit = objective_func_for_flow(v_max_scan_boundary)
        if abs(obj_at_limit) < xtol_brentq * 10:
            s_net_check = v_max_scan_boundary / target_Q_veh_per_sec - l_veh
            if s_net_check >= s0 - xtol_brentq*100 or (abs(v_max_scan_boundary)<xtol_brentq and abs(s_net_check-s0)<xtol_brentq*100) : # Допуск для s0
                if s_net_check >= -xtol_brentq : # Допускаем s_net немного отрицательным из-за xtol
                     s_net_final = max(0, s_net_check)
                     if not any(abs(v_max_scan_boundary - r[0]) < xtol_brentq*10 for r in results):
                        results.append((v_max_scan_boundary, s_net_final))
                        if verbose: print(f"Найдено решение на границе v_e={v_max_scan_boundary:.3f}, s_e_net={s_net_final:.3f} для Q={target_Q_veh_per_sec:.4f}")
        return results


    scan_points_v = np.linspace(v_min_scan_boundary, v_max_scan_boundary, num_scan_intervals + 1)
    
    f_prev = objective_func_for_flow(scan_points_v[0])

    for i in range(num_scan_intervals):
        v_low = scan_points_v[i]
        v_high = scan_points_v[i+1]
        f_curr = objective_func_for_flow(v_high)

        if f_prev * f_curr <= 0: # Смена знака или корень на границе
            try:
                # Проверим, не слишком ли мал интервал
                if abs(v_high - v_low) < xtol_brentq:
                    if abs(f_prev) < xtol_brentq * 10 : # Корень на v_low
                        v_e_sol = v_low
                    elif abs(f_curr) < xtol_brentq * 10 : # Корень на v_high
                        v_e_sol = v_high
                    else: # Интервал слишком мал, но нет явного корня на границе
                        f_prev = f_curr
                        continue
                else:
                    v_e_sol = brentq(objective_func_for_flow, v_low, v_high, xtol=xtol_brentq, maxiter=maxiter_brentq)
                
                s_e_total_sol = v_e_sol / target_Q_veh_per_sec
                s_e_net_sol = s_e_total_sol - l_veh

                # Проверка валидности решения
                valid_solution = True
                if s_e_net_sol < -xtol_brentq : # Чистый зазор не может быть отрицательным (с допуском)
                    valid_solution = False
                    if verbose: print(f"Отброшено (s_net<0): v_e={v_e_sol:.3f}, s_e_net={s_e_net_sol:.3f}")
                
                # Если скорость близка к нулю, чистый зазор должен быть s0
                if abs(v_e_sol) < xtol_brentq * 10 and abs(s_e_net_sol - s0) > xtol_brentq * 100:
                    # Это может быть артефакт brentq, если objective_func_for_flow не идеальна около v=0
                    # Попробуем скорректировать s_e_net_sol к s0 если accel(s0,0,0) ~ 0
                    accel_s0_v0 = calculate_idm_acceleration(s0, 0, 0.0, params)
                    if abs(accel_s0_v0) < xtol_brentq * 10 :
                        if verbose: print(f"Коррекция: для v_e~0 ({v_e_sol:.3f}), s_e_net ({s_e_net_sol:.3f}) был {s_e_net_sol:.3f}, но приведен к s0 ({s0})")
                        s_e_net_sol = s0
                    else: # Если (0,s0) не равновесие, то и это не должно быть
                        valid_solution = False
                        if verbose: print(f"Отброшено (v_e~0, s_net!=s0, (0,s0) не равновесие): v_e={v_e_sol:.3f}, s_e_net={s_e_net_sol:.3f}")


                # Если скорость положительна, чистый зазор должен быть не меньше s0 (с допуском)
                if v_e_sol > xtol_brentq * 10 and s_e_net_sol < s0 - xtol_brentq * 100 :
                    valid_solution = False
                    if verbose: print(f"Отброшено (v_e>0, s_net < s0): v_e={v_e_sol:.3f}, s_e_net={s_e_net_sol:.3f} (s0={s0})")

                if valid_solution:
                    s_e_net_sol = max(0, s_e_net_sol) # Гарантируем неотрицательность после всех проверок
                    # Проверка на уникальность (чтобы не добавлять один и тот же корень много раз)
                    is_unique = True
                    for r_v, r_s in results:
                        if abs(r_v - v_e_sol) < xtol_brentq * 10: # Если скорости очень близки
                            is_unique = False
                            break
                    if is_unique:
                        results.append((v_e_sol, s_e_net_sol))
                        if verbose: print(f"Найдено решение сканированием: v_e={v_e_sol:.3f}, s_e_net={s_e_net_sol:.3f} для Q={target_Q_veh_per_sec:.4f} в [{v_low:.3f}, {v_high:.3f}]")
            
            except ValueError: # brentq не смог найти корень (например, f_prev*f_curr > 0 из-за численных неточностей)
                if verbose: print(f"ValueError в brentq для Q={target_Q_veh_per_sec:.4f} в [{v_low:.3f}, {v_high:.3f}] (f_prev={f_prev:.2e}, f_curr={f_curr:.2e})")
                pass
            except RuntimeError as e: # brentq столкнулся с проблемой
                 if verbose: print(f"RuntimeError в brentq для Q={target_Q_veh_per_sec:.4f}: {e}")
                 pass
        
        f_prev = f_curr
    
    # Сортировка результатов по v_e
    results.sort(key=lambda x: x[0])
    if verbose and not results: print(f"Решения для Q={target_Q_veh_per_sec:.4f} не найдены.")
    elif verbose and results: print(f"Всего найдено {len(results)} решений для Q={target_Q_veh_per_sec:.4f}.")
    return results

def calculate_partial_derivatives(s_star_net, v_star, params, tol=1e-6):
    """
    Вычисляет частные производные f_s, f_dv, f_v в точке равновесия (s_star_net, 0, v_star).
    s_star_net: равновесный чистый зазор.
    Производная f_s берется по чистому зазору s_star_net.
    """
    a_max = params['a_max_accel']
    s0 = params['s0_jam_distance']
    T = params['T_safe_time_headway']
    v0_target = params['v0_desired_speed']
    delta = params['delta_accel_exponent']
    b_comfort_decel = params['b_comfort_decel'] 

    if v0_target < 0: 
        return float('nan'), float('nan'), float('nan')
    if s_star_net < 0 : 
        return float('nan'), float('nan'), float('nan')

    # Проверки на физичность точки для расчета производных
    if v_star > tol and s_star_net < s0 - tol :
        return float('nan'), float('nan'), float('nan')
    if abs(v_star) < tol and abs(s_star_net - s0) > tol * 10 : # Увеличенный допуск для s0 при v=0
        # Если v=0, но s_star_net сильно отличается от s0, это не равновесие IDM.
        # Производные можно посчитать, но их интерпретация для устойчивости сомнительна.
        # Если s_star_net < s0 - tol при v_star=0, это тоже нефизично.
        if s_star_net < s0 - tol : return float('nan'), float('nan'), float('nan')
        # Если s_star_net > s0 + tol*10 и v_star=0, это точка не в равновесии IDM.
        # print(f"Warning: Calculating derivatives at non-equilibrium point (v_star~0, s_star_net ({s_star_net:.2f}) != s0 ({s0:.2f}))")


    s_hat_eq = calculate_s_hat(v_star, 0, params)

    # f_s = производная по s_star_net
    try:
        if s_star_net <= 1e-6 : 
            f_s = float('inf') if s_hat_eq > 1e-6 else 0.0 
        else:
            f_s = 2 * a_max * (s_hat_eq**2) / (s_star_net**3)
    except ZeroDivisionError: 
        f_s = float('inf') if a_max * (s_hat_eq**2) > 1e-9 else 0.0

    # f_dv
    sqrt_ab_val = a_max * b_comfort_decel 
    if sqrt_ab_val <= 1e-9 or v_star < -tol: 
        f_dv = 0.0 
    else:
        if s_star_net <= 1e-6: 
             f_dv = float('inf') if a_max * s_hat_eq * v_star > 1e-9 else 0.0
        else:
            denominator_f_dv = (s_star_net**2) * math.sqrt(sqrt_ab_val)
            if abs(denominator_f_dv) < 1e-9 : 
                f_dv = float('inf') if a_max * s_hat_eq * v_star > 1e-9 else 0.0 
            else:
                # v_star в s_hat для члена с dv - это скорость текущего авто, она положительна в равновесии.
                f_dv = a_max * s_hat_eq * v_star / denominator_f_dv 

    # f_v
    term_v_ratio_deriv = 0.0
    if abs(v0_target) < tol: 
        if delta > 1 + tol and v_star > tol : term_v_ratio_deriv = -float('inf') 
        elif abs(delta - 1) < tol and v_star > tol : term_v_ratio_deriv = -float('inf') 
        else: term_v_ratio_deriv = 0.0 
    elif v_star < -tol:
        term_v_ratio_deriv = float('nan')
    elif abs(v_star) < tol : 
        if abs(delta - 1) < tol: term_v_ratio_deriv = -delta / v0_target 
        elif delta < 1: term_v_ratio_deriv = -float('inf') 
        else: term_v_ratio_deriv = 0.0 
    else: 
        v_eff_for_deriv = v_star 
        try:
            term_v_ratio_deriv = -delta / v0_target * (v_eff_for_deriv / v0_target)**(delta - 1)
        except (ValueError, ZeroDivisionError): 
            term_v_ratio_deriv = float('nan')


    term_spacing_deriv_vs_v = 0.0
    if s_star_net <= 1e-6 : 
        term_spacing_deriv_vs_v = -float('inf') if (a_max * 2 * s_hat_eq * T) > 0 else (float('inf') if (a_max * 2 * s_hat_eq * T) < 0 else 0.0)
    else:
        d_shat_dv = T # так как dv=0 в точке равновесия
        term_spacing_deriv_vs_v = -a_max * (2 * s_hat_eq / (s_star_net**2)) * d_shat_dv
        
    f_v_val = a_max * term_v_ratio_deriv + term_spacing_deriv_vs_v

    # Обработка NaN/Inf из-за компонентов
    if math.isnan(term_v_ratio_deriv) or math.isnan(term_spacing_deriv_vs_v): 
        f_v = float('nan')
    elif math.isinf(term_v_ratio_deriv) and math.isinf(term_spacing_deriv_vs_v) and \
         ((a_max * term_v_ratio_deriv > 0) != (term_spacing_deriv_vs_v > 0)) : # inf - inf
        f_v = float('nan') 
    elif math.isinf(a_max * term_v_ratio_deriv) and not math.isinf(term_spacing_deriv_vs_v) :
        f_v = a_max * term_v_ratio_deriv
    elif math.isinf(term_spacing_deriv_vs_v) and not math.isinf(a_max*term_v_ratio_deriv):
        f_v = term_spacing_deriv_vs_v
    elif math.isinf(a_max*term_v_ratio_deriv) and math.isinf(term_spacing_deriv_vs_v) : # оба inf одного знака
        f_v = a_max*term_v_ratio_deriv # или term_spacing_deriv_vs_v, если они одного знака
    else:
        f_v = f_v_val
    
    return f_s, f_dv, f_v

def check_rational_driving_constraints(f_s, f_dv, f_v):
    valid_fs = f_s > 1e-9 
    valid_fdv = f_dv >= -1e-9 # Может быть 0, если v_star=0 или T=0 и s0=0 в s_hat
    valid_fv = f_v < -1e-9  
    return valid_fs and valid_fdv and valid_fv

def analyze_platoon_stability(f_s, f_dv, f_v, verbose=True):
    if verbose: print("--- Анализ устойчивости взвода (Platoon Stability) ---")
    if any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]):
        if verbose: print("Невозможно проанализировать: одна из производных NaN или Inf.")
        return False
    
    # Условия Рауса-Гурвица для хар. ур. k^2 + (f_dv - f_v)k + f_s = 0
    # 1. Все коэффициенты > 0
    #    a0 = 1 > 0 (всегда)
    #    a1 = f_dv - f_v > 0
    #    a2 = f_s > 0
    # Это и есть критерии устойчивости.
    
    coeff_b = f_dv - f_v # Должен быть > 0
    coeff_c = f_s      # Должен быть > 0
    is_stable = False

    if coeff_b > 1e-9 and coeff_c > 1e-9:
        is_stable = True
        if verbose: print(f"(f_Δv-f_v)={coeff_b:.4f} > 0, f_s={coeff_c:.4f} > 0. Раус-Гурвиц: УСТОЙЧИВ.")
    else:
        if verbose: print(f"(f_Δv-f_v)={coeff_b:.4f}, f_s={coeff_c:.4f}. Раус-Гурвиц не выполнен. Анализ корней:")
        # Если Р-Г не выполнен, это не всегда значит неустойчивость, если какой-то коэфф. = 0.
        # Но если <0, то неустойчив.
        # Проверим корни напрямую: l = [-coeff_b +/- sqrt(coeff_b^2 - 4*coeff_c)] / 2
        # Устойчивость если Re(l) < 0 для всех корней.
        # Это эквивалентно coeff_b > 0 и coeff_c > 0 (если корни вещественные или комплексные с отр. реальной частью).
        try:
            D = coeff_b**2 - 4*coeff_c # Дискриминант
            if D >= 0: # Вещественные корни
                l1=(-coeff_b + math.sqrt(D))/2
                l2=(-coeff_b - math.sqrt(D))/2
                if verbose: print(f"Вещественные корни: λ1={l1:.4f}, λ2={l2:.4f}")
                is_stable = (l1 < -1e-9 and l2 < -1e-9)
            else: # Комплексные корни
                real_part = -coeff_b/2
                # imag_part = math.sqrt(-D)/2
                if verbose: print(f"Комплексные корни: Re(λ)={real_part:.4f}")
                is_stable = (real_part < -1e-9)
            
            if is_stable and verbose: print("Результат анализа корней: Взвод УСТОЙЧИВ.")
            elif not is_stable and verbose: print("Результат анализа корней: Взвод НЕУСТОЙЧИВ.")
        
        except (OverflowError, ValueError): 
            is_stable=False # Если ошибка в вычислении корней
            if verbose: print("Ошибка при анализе корней для устойчивости взвода.")
    
    return is_stable


def analyze_string_stability(f_s, f_dv, f_v, verbose=True):
    if verbose: print("--- Анализ устойчивости цепочки (String Stability) ---")
    if any(math.isnan(x) or math.isinf(x) for x in [f_s,f_dv,f_v]):
        if verbose: print("NaN/Inf в производных, невозможно проанализировать устойчивость цепочки.")
        return False, float('nan') # Возвращаем состояние и значение K
        
    # Критерий из статьи (Eq. 20 в предположении f_v < 0):
    # λ₂ = (f_s / f_v³) * (f_v²/2 - f_Δv*f_v - f_s)
    # Устойчивость если Re(λ₂) < 0.
    # Поскольку f_s > 0 (рац. вождение) и f_v < 0 (рац. вождение), то f_s/f_v³ < 0.
    # Значит, для Re(λ₂) < 0 нужно, чтобы (f_v²/2 - f_Δv*f_v - f_s) > 0.
    # Обозначим K = f_v²/2 - f_Δv*f_v - f_s.
    # Условие устойчивости цепочки: K > 0 (при выполнении условий рац. вождения f_s>0, f_v<0).

    K_condition_val = float('nan')
    is_stable = False

    # Проверяем условия рационального вождения для интерпретации K
    # f_s > 0 и f_v < 0 являются ключевыми. f_dv >= 0 тоже желательно.
    rational_fs = f_s > 1e-9
    rational_fv = f_v < -1e-9
    # rational_fdv = f_dv >= -1e-9 # f_dv может быть близко к 0

    if not (rational_fs and rational_fv):
        if verbose: 
            print(f"Условия рационального вождения не выполнены для анализа K (f_s={f_s:.3f}, f_v={f_v:.3f}).")
        # K все равно можно посчитать, но его интерпретация как критерия устойчивости под вопросом.
        try:
            K_condition_val = (f_v**2)/2 - f_dv*f_v - f_s
        except (OverflowError, ValueError):
             K_condition_val = float('nan')
        return False, K_condition_val # Неустойчива или неопределена из-за нарушения рац. вождения

    try:
        K_condition_val = (f_v**2)/2 - f_dv*f_v - f_s
        if K_condition_val > 1e-9 : # K > 0
            is_stable = True
        
        if verbose: 
            print(f"K = f_v^2/2 - f_Δv*f_v - f_s = {K_condition_val:.4f}")
            print(f"Результат: Цепочка {'УСТОЙЧИВА' if is_stable else 'НЕУСТОЙЧИВА'}.")
            
    except (OverflowError, ValueError): 
        is_stable=False 
        K_condition_val = float('nan')
        if verbose: print("Ошибка вычисления K для устойчивости цепочки.")
        
    return is_stable, K_condition_val

def collect_data_for_plots(s_star_net_range, params):
    s_values, v_values, fs_values, fdv_values, fv_values, k_values, platoon_flags, string_flags = [],[],[],[],[],[],[],[]
    for s_star_n in s_star_net_range:
        v_star = find_equilibrium_velocity(s_star_n, params)
        if v_star is None or math.isnan(v_star) or v_star < 0: # Скорость не может быть отрицательной
             # print(f"s_net={s_star_n}, v_star is invalid: {v_star}")
            continue
            
        f_s, f_dv, f_v = calculate_partial_derivatives(s_star_n, v_star, params)
        if any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]): 
            # print(f"s_net={s_star_n}, v_star={v_star}, derivs invalid: fs={f_s},fdv={f_dv},fv={f_v}")
            continue
            
        s_values.append(s_star_n)
        v_values.append(v_star)
        fs_values.append(f_s)
        fdv_values.append(f_dv)
        fv_values.append(f_v)
        
        rational = check_rational_driving_constraints(f_s, f_dv, f_v)
        
        platoon_stable = analyze_platoon_stability(f_s, f_dv, f_v, verbose=False) if rational else False
        string_stable, K = analyze_string_stability(f_s, f_dv, f_v, verbose=False)
        
        platoon_flags.append(platoon_stable)
        string_flags.append(string_stable if rational else False) # Строгая уст-ть только при рац. вождении
        k_values.append(K) # Сохраняем K для графика

    return {
        's_star_net':np.array(s_values), 
        'v_star':np.array(v_values), 
        'f_s':np.array(fs_values), 
        'f_dv':np.array(fdv_values), 
        'f_v':np.array(fv_values), 
        'K_condition':np.array(k_values), 
        'platoon_stable':np.array(platoon_flags), 
        'string_stable':np.array(string_flags)
    }

def plot_stability_analysis(data, params):
    # data['s_star_net'] теперь содержит чистый зазор
    if len(data['s_star_net']) == 0: 
        print("Нет данных для построения графиков s_star_net vs v*.")
        return
        
    v_star_kmh = data['v_star'] * 3.6
    s_star_net_plot = data['s_star_net']
    
    plt.figure(figsize=(12, 8))
    
    # 1. Фундаментальная диаграмма (v* от s_star_net)
    plt.subplot(2,2,1)
    plt.plot(s_star_net_plot, v_star_kmh, 'b.-')
    plt.xlabel('Равновесный чистый зазор s*_net (м)')
    plt.ylabel('Равновесная скорость v* (км/ч)')
    plt.title('Фундаментальная диаграмма (v* от s*_net)')
    plt.grid(True)
    
    # Сортировка по v* для остальных графиков, если v* используется как ось X
    idx_sorted_by_v = np.argsort(data['v_star'])
    v_star_kmh_sorted = v_star_kmh[idx_sorted_by_v]
    
    # 2. Критерий K от v*
    ax_k = plt.subplot(2,2,2)
    k_condition_sorted = data['K_condition'][idx_sorted_by_v]
    plt.plot(v_star_kmh_sorted, k_condition_sorted, 'r.-')
    plt.axhline(0, color='k', lw=0.8, ls='--')
    plt.xlabel('Равновесная скорость v* (км/ч)')
    plt.ylabel('Критерий K')
    plt.title('Критерий K от v*')
    plt.grid(True)
    
    # 3. Производные от v*
    plt.subplot(2,2,3)
    plt.plot(v_star_kmh_sorted, data['f_s'][idx_sorted_by_v], marker='.', linestyle='-', label='f_s (по s_net)') 
    plt.plot(v_star_kmh_sorted, data['f_dv'][idx_sorted_by_v], marker='.', linestyle='-', label='f_Δv') 
    plt.plot(v_star_kmh_sorted, data['f_v'][idx_sorted_by_v], marker='.', linestyle='-', label='f_v') 
    plt.xlabel('Равновесная скорость v* (км/ч)')
    plt.ylabel('Значения производных')
    plt.title('Частные производные от v*')
    plt.legend()
    plt.grid(True)
    
    # 4. Области устойчивости от v*
    ax_stab = plt.subplot(2,2,4)
    platoon_stable_sorted = data['platoon_stable'][idx_sorted_by_v]
    string_stable_sorted = data['string_stable'][idx_sorted_by_v]
    
    # Отфильтруем NaN в v_star_kmh_sorted для scatter plot
    valid_indices_for_scatter = ~np.isnan(v_star_kmh_sorted)
    v_scatter = v_star_kmh_sorted[valid_indices_for_scatter]
    ps_scatter = platoon_stable_sorted[valid_indices_for_scatter]
    ss_scatter = string_stable_sorted[valid_indices_for_scatter]

    if np.any(ps_scatter): 
        ax_stab.scatter(v_scatter[ps_scatter], np.ones(np.sum(ps_scatter))*1.0, c='c', marker='o', label='Взвод Уст.', alpha=0.7)
    if np.any(~ps_scatter): 
        ax_stab.scatter(v_scatter[~ps_scatter], np.ones(np.sum(~ps_scatter))*1.0, c='m', marker='x', label='Взвод Неуст.', alpha=0.7)
    
    if np.any(ss_scatter): 
        ax_stab.scatter(v_scatter[ss_scatter], np.ones(np.sum(ss_scatter))*0.5, c='g', marker='o', label='Цеп. Уст.', alpha=0.7)
    if np.any(~ss_scatter): 
        ax_stab.scatter(v_scatter[~ss_scatter], np.ones(np.sum(~ss_scatter))*0.5, c='r', marker='x', label='Цеп. Неуст.', alpha=0.7)
    
    ax_stab.set_yticks([0.5,1.0])
    ax_stab.set_yticklabels(['Цепочка','Взвод'])
    ax_stab.set_xlabel('Равновесная скорость v* (км/ч)')
    ax_stab.set_title('Области устойчивости')
    ax_stab.set_ylim(0,1.5)
    ax_stab.legend(fontsize='small')
    ax_stab.grid(True)
    
    param_summary = f"a={params['a_max_accel']:.2f}, T={params['T_safe_time_headway']:.2f}, b={params['b_comfort_decel']:.2f}, s0={params['s0_jam_distance']:.2f}, L={params['l_vehicle_length']:.2f}"
    plt.suptitle(f"Анализ IDM (на основе чистого зазора s*_net)\\nПараметры: {param_summary}", fontsize=11)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


def collect_data_for_param_sweep(
    param_to_sweep_key, sweep_values, base_idm_params, 
    fixed_s_star_net=None, fixed_v_star=None, fixed_Q=None, 
    verbose=False):
    
    param_vals_list, s_star_net_list, v_star_list, K_list, platoon_stable_list, string_stable_list = [],[],[],[],[],[]

    num_fixed_conditions = sum(x is not None for x in [fixed_s_star_net, fixed_v_star, fixed_Q])
    if num_fixed_conditions == 0:
        raise ValueError("Необходимо зафиксировать s_star_net, v_star или Q.")
    if num_fixed_conditions > 1:
        raise ValueError("Можно зафиксировать только одну переменную: s_star_net, v_star или Q.")

    for val in sweep_values:
        current_params = base_idm_params.copy()
        current_params[param_to_sweep_key] = val
        
        equilibria_to_process = [] # Список пар (s_net, v)

        if fixed_s_star_net is not None:
            current_s_net = fixed_s_star_net
            current_v = find_equilibrium_velocity(current_s_net, current_params)
            if current_v is not None and not (math.isnan(current_v) or math.isinf(current_v) or current_v < -1e-6):
                equilibria_to_process.append({'s_net': current_s_net, 'v': current_v})
        elif fixed_v_star is not None:
            current_v = fixed_v_star
            current_s_net = calculate_s_star_for_fixed_v_star(current_v, current_params)
            if current_s_net is not None and not (math.isnan(current_s_net) or math.isinf(current_s_net)):
                 equilibria_to_process.append({'s_net': current_s_net, 'v': current_v})
        elif fixed_Q is not None:
            # find_all_equilibrium_states_for_flow возвращает список [(v_e, s_e_net), ...]
            found_states = find_all_equilibrium_states_for_flow(fixed_Q, current_params, verbose=verbose)
            for v_eq, s_net_eq in found_states:
                # Дополнительная проверка валидности перед добавлением
                if s_net_eq is not None and not (math.isnan(s_net_eq) or math.isinf(s_net_eq)) and \
                   v_eq is not None and not (math.isnan(v_eq) or math.isinf(v_eq) or v_eq < -1e-6):
                    equilibria_to_process.append({'s_net': s_net_eq, 'v': v_eq})
                elif verbose:
                    print(f"Отброшено состояние (при Q={fixed_Q:.3f}) для {param_to_sweep_key}={val}: s*_net={s_net_eq}, v*={v_eq}")
        
        if not equilibria_to_process and verbose:
            print(f"Для {param_to_sweep_key}={val}: не найдено валидных равновесных состояний.")
            # continue # Не пропускаем, чтобы param_vals_list был синхронизирован, если нужно позже заполнять NaN

        for eq_state in equilibria_to_process:
            current_s_net_eq = eq_state['s_net']
            current_v_eq = eq_state['v']

            f_s, f_dv, f_v = calculate_partial_derivatives(current_s_net_eq, current_v_eq, current_params)
            
            if any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]):
                if verbose: print(f"Пропуск {param_to_sweep_key}={val} (s_net={current_s_net_eq:.2f}, v={current_v_eq:.2f}): NaN/Inf в производных")
                # Заполняем пустыми значениями, чтобы сохранить соответствие с param_values
                param_vals_list.append(val)
                s_star_net_list.append(current_s_net_eq) # или float('nan') если хотим четко пометить
                v_star_list.append(current_v_eq)       # или float('nan')
                K_list.append(float('nan'))
                platoon_stable_list.append(False) # или None/NaN
                string_stable_list.append(False)  # или None/NaN
                continue
            
            param_vals_list.append(val)
            s_star_net_list.append(current_s_net_eq)
            v_star_list.append(current_v_eq)
            
            rational = check_rational_driving_constraints(f_s, f_dv, f_v)
            platoon_stable = analyze_platoon_stability(f_s, f_dv, f_v, verbose=False) if rational else False
            string_stable, K = analyze_string_stability(f_s, f_dv, f_v, verbose=False)
            
            K_list.append(K if rational else float('nan')) # K имеет смысл только при рац. вождении для анализа уст-ти
            platoon_stable_list.append(platoon_stable)
            string_stable_list.append(string_stable if rational else False) 
        
    return {
        'param_values': np.array(param_vals_list), 
        's_star_net': np.array(s_star_net_list), 
        'v_star': np.array(v_star_list), 
        'K_condition': np.array(K_list),
        'platoon_stable': np.array(platoon_stable_list), 
        'string_stable': np.array(string_stable_list)
    }

def plot_stability_for_parameter_sweep(data, swept_param_key, swept_param_label, fixed_condition_label, base_params):
    if len(data['param_values']) == 0:
        print(f"Нет данных для построения графиков для параметра {swept_param_label}.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    param_x_values = data['param_values']
    
    # Фильтруем NaN значения перед построением scatter, чтобы избежать предупреждений и ошибок
    valid_indices_vs = ~np.isnan(param_x_values) & ~np.isnan(data['v_star']) & ~np.isnan(data['s_star_net'])
    valid_indices_k = ~np.isnan(param_x_values) & ~np.isnan(data['K_condition'])

    # Верхний график: v* и s*_net (чистый зазор)
    axs0_twin = axs[0].twinx()
    
    # Используем scatter вместо plot
    sc1 = axs[0].scatter(param_x_values[valid_indices_vs], data['v_star'][valid_indices_vs] * 3.6, marker='o', s=20, alpha=0.7, color='blue', label='v* (км/ч)')
    sc2 = axs0_twin.scatter(param_x_values[valid_indices_vs], data['s_star_net'][valid_indices_vs], marker='x', s=20, alpha=0.7, color='green', label='s*_net (м)')
    
    axs[0].set_ylabel('Равновесная скорость v* (км/ч)', color='blue') # sc1.get_facecolor() может вернуть массив
    axs0_twin.set_ylabel('Равновесный чистый зазор s*_net (м)', color='green') # sc2.get_facecolor()
    axs[0].tick_params(axis='y', labelcolor='blue')
    axs0_twin.tick_params(axis='y', labelcolor='green')
    
    # Создание легенды для scatter plots
    # axs[0].legend(handles=[sc1, sc2], loc='best') # Прямое использование sc1, sc2 для legend
    # Альтернативный способ для scatter, если handles не работает прямо
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='v* (км/ч)', markersize=5, markerfacecolor='blue'),
                       Line2D([0], [0], marker='x', color='w', label='s*_net (м)', markersize=5, markerfacecolor='green')]
    axs[0].legend(handles=legend_elements, loc='best')

    axs[0].set_title(f'Зависимость v* и s*_net от {swept_param_label}')
    axs[0].grid(True)


    # Средний график: Критерий K
    # Используем scatter вместо plot
    axs[1].scatter(param_x_values[valid_indices_k], data['K_condition'][valid_indices_k], marker='.', s=20, color='r', alpha=0.7)
    axs[1].axhline(0, color='black', lw=0.8, linestyle='--')
    axs[1].set_ylabel('Критерий K')
    axs[1].set_title(f'Критерий устойчивости цепочки K vs. {swept_param_label}')
    axs[1].grid(True)

    # Нижний график: Области устойчивости (уже использует scatter)
    ps_mask = data['platoon_stable']
    ss_mask = data['string_stable']
    valid_indices = ~np.isnan(param_x_values) & ~np.isnan(data['K_condition']) 
    
    px_valid = param_x_values[valid_indices]
    ps_mask_valid = ps_mask[valid_indices]
    ss_mask_valid = ss_mask[valid_indices]

    if np.any(ps_mask_valid): axs[2].scatter(px_valid[ps_mask_valid], np.ones(np.sum(ps_mask_valid))*1.0, c='c', marker='o', label='Взвод Уст.', alpha=0.7)
    if np.any(~ps_mask_valid): axs[2].scatter(px_valid[~ps_mask_valid], np.ones(np.sum(~ps_mask_valid))*1.0, c='m', marker='x', label='Взвод Неуст.', alpha=0.7)
    if np.any(ss_mask_valid): axs[2].scatter(px_valid[ss_mask_valid], np.ones(np.sum(ss_mask_valid))*0.5, c='g', marker='o', label='Цеп. Уст.', alpha=0.7)
    if np.any(~ss_mask_valid): axs[2].scatter(px_valid[~ss_mask_valid], np.ones(np.sum(~ss_mask_valid))*0.5, c='r', marker='x', label='Цеп. Неуст.', alpha=0.7)
    
    axs[2].set_yticks([0.5,1.0])
    axs[2].set_yticklabels(['Цепочка','Взвод'])
    axs[2].set_xlabel(swept_param_label)
    axs[2].set_title('Области устойчивости')
    axs[2].set_ylim(0,1.5)
    axs[2].legend(fontsize='small')
    axs[2].grid(True)

    param_details_list = []
    for k_param, v_param in DEFAULT_IDM_PARAMS.items():
        if k_param == swept_param_key: continue
        current_val = base_params.get(k_param, v_param)
        # Форматируем L и s0 с одним знаком после запятой, остальные с двумя
        if k_param in ['l_vehicle_length', 's0_jam_distance']:
            param_details_list.append(f"{k_param.split('_')[0]}={current_val:.1f}")
        else:
            param_details_list.append(f"{k_param.split('_')[0]}={current_val:.2f}")

    param_details = ", ".join(param_details_list)
    
    fig.suptitle(f'Анализ уст-ти IDM: {swept_param_label} ({fixed_condition_label})\\nОст. параметры: {param_details}', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

if __name__ == '__main__':
    params = DEFAULT_IDM_PARAMS.copy()
    # params['s0_jam_distance'] = 0.1 # Пример изменения s0 для теста
    
    # Пример 1: Анализ для ОДНОЙ точки, задавая чистый зазор s_star_net
    s_star_net_example = 5.0 # Пример чистого зазора
    print(f"--- Анализ для ОДНОЙ точки: s*_net = {s_star_net_example} м ---")
    v_star = find_equilibrium_velocity(s_star_net_example, params)
    if v_star is not None and not math.isnan(v_star) and v_star >=0:
        print(f"Равновесная скорость v* = {v_star:.2f} м/с ({(v_star*3.6):.2f} км/ч)")
        f_s,f_dv,f_v = calculate_partial_derivatives(s_star_net_example,v_star,params)
        if not any(math.isnan(x) or math.isinf(x) for x in [f_s,f_dv,f_v]):
            if check_rational_driving_constraints(f_s,f_dv,f_v):
                analyze_platoon_stability(f_s,f_dv,f_v, verbose=True)
                string_stable, K = analyze_string_stability(f_s,f_dv,f_v, verbose=True)
            else:
                print("Условия рац. вождения не выполнены, анализ уст-ти не проводится.")
        else: print("Производные не определены.")
    else: print(f"Равновесная скорость не найдена или невалидна для s*_net={s_star_net_example} (v*={v_star}).")
    print("-"*50)

    # Пример 2: Графики: Зависимость от s_star_net (чистый зазор)
    print("\n--- Графики: Зависимость от s*_net (чистый зазор) ---")
    # Диапазон для чистого зазора s_star_net. Начинаем от s0.
    min_s_star_net = params['s0_jam_distance'] 
    max_s_star_net = 150 
    s_star_net_values_for_plot = np.linspace(min_s_star_net, max_s_star_net, 100)
    # Добавим точку s0 + epsilon, чтобы лучше видеть поведение около s0
    s_star_net_values_for_plot = np.sort(np.unique(np.concatenate(([params['s0_jam_distance'] + 1e-3], s_star_net_values_for_plot))))

    plot_data_s_v = collect_data_for_plots(s_star_net_values_for_plot, params.copy())
    if plot_data_s_v and len(plot_data_s_v['s_star_net']) > 0 :
        plot_stability_analysis(plot_data_s_v, params.copy())
    else:
        print("Не удалось собрать данные для графика s*_net vs v*.")
    print("-"*50)

    # Пример 3: Варьирование T (время реакции) при фиксированном s_star_net (чистый зазор)
    print("\n--- Графики: Варьирование T (время реакции) при фикс. s*_net ---")
    fixed_s_net_for_T_sweep = 10.0 # Фиксированный чистый зазор
    T_sweep_values = np.linspace(0.5, 3.0, 50)
    data_sweep_T_fixed_s_net = collect_data_for_param_sweep(
        param_to_sweep_key='T_safe_time_headway', sweep_values=T_sweep_values,
        base_idm_params=params.copy(), fixed_s_star_net=fixed_s_net_for_T_sweep, verbose=False
    )
    if data_sweep_T_fixed_s_net and len(data_sweep_T_fixed_s_net['param_values']) > 0:
        plot_stability_for_parameter_sweep(
            data_sweep_T_fixed_s_net, swept_param_key='T_safe_time_headway',
            swept_param_label='Время реакции T (с)', 
            fixed_condition_label=f"s*_net = {fixed_s_net_for_T_sweep:.1f} м", # Обновлено
            base_params=params.copy()
        )
    else:
         print(f"Не удалось собрать данные для варьирования T при s*_net={fixed_s_net_for_T_sweep:.1f} м.")
    print("-"*50)

    # Пример 4: Варьирование 'a' (макс. ускорение) при фиксированной v*
    print("\n--- Графики: Варьирование 'a' (макс. ускорение) при фикс. v* ---")
    fixed_v_for_a_sweep = 20.0  # м/с
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
    else:
        print(f"Не удалось собрать данные для варьирования 'a' при v*={fixed_v_for_a_sweep:.1f} м/с.")
    print("-"*50)
    
    # Пример 5: Варьирование T (время реакции) при фиксированном Q (поток)
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
            fixed_condition_label=f"Q = {target_Q_veh_per_hour:.0f} авто/час ({target_Q_veh_per_sec:.3f} авто/с)",
            base_params=params.copy()
        )
    else:
        print(f"Не удалось собрать данные для варьирования T при Q={target_Q_veh_per_hour:.0f} авто/час.")

    print("\nАнализ завершен.") 