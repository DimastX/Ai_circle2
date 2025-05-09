import math
import numpy as np
import matplotlib.pyplot as plt

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
        # Формула для s1 != 0 (более общая, но может вызвать проблемы при v=0)
        # s_hat_val = params['s0_jam_distance'] + \
        #             params['s1_gap_param'] * math.sqrt(max(0, v / params['v0_desired_speed'])) + \
        #             params['T_safe_time_headway'] * v - \
        #             (v * dv) / (2 * math.sqrt(params['a_max_accel'] * params['b_comfort_decel']))
        # Для простоты и соответствия статье, используем s1=0
        print("Предупреждение: Эта реализация оптимизирована для s1_gap_param = 0.")

    # Проверка на случай нулевого значения под корнем, если a_max_accel или b_comfort_decel равны нулю
    sqrt_term_val = params['a_max_accel'] * params['b_comfort_decel']
    if sqrt_term_val <= 0: # Не должно быть для стандартных параметров, но для общности
        # print("Предупреждение: Произведение a_max_accel * b_comfort_decel <= 0. Невозможно вычислить корень.")
        # В таком случае, член с dv не определен. Поведение зависит от контекста.
        # Для анализа устойчивости, где dv=0, это не повлияет на s_hat_eq.
        # Если dv != 0, это проблема. Здесь можно вернуть ошибку или обработать.
        # Для текущей реализации, когда dv=0 в find_equilibrium_velocity, это не вызовет проблему.
        # Но для calculate_idm_acceleration с dv != 0 это может быть проблемой.
        # Вернем значение, как если бы член с dv был 0, если сам dv не 0.
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
    if s <= params['l_vehicle_length']: # Чистая дистанция s - l не может быть отрицательной
        net_spacing = s - params['l_vehicle_length']
        if net_spacing <= 1e-3: 
            return -params['b_comfort_decel'] * 10 # Сильное торможение

    # Проверка params['v0_desired_speed'] > 0
    if params['v0_desired_speed'] <= 0:
        # print("Предупреждение: v0_desired_speed <= 0. Ускорение не определено корректно.")
        term_vel = 1.0 # Чтобы (1 - term_vel ...) дало торможение или 0
    else:
        term_vel = (max(0,v) / params['v0_desired_speed'])**params['delta_accel_exponent']

    s_hat = calculate_s_hat(v, dv, params)
        
    net_clearance = s - params['l_vehicle_length']
    if net_clearance <= 1e-3: # Изменено с 0 на малое положительное число для предотвращения деления на 0
        # Если зазор очень мал или отрицателен (что не должно быть для s > l)
        # Этот случай должен быть обработан выше, но для надежности:
        term_spacing = (max(0, params['s0_jam_distance']) / (1e-3 if net_clearance <=0 else net_clearance) )**2 
    else:
        term_spacing = (s_hat / net_clearance)**2 # s_hat может быть отрицательным если dv очень большое
                                                # но (s_hat/net_clearance)^2 всегда >= 0
        
    acceleration = params['a_max_accel'] * (1 - term_vel - term_spacing)
    return acceleration

def find_equilibrium_velocity(s_star, params, tol=1e-6, max_iter=100):
    """
    Находит равновесную скорость v* для заданной равновесной дистанции s*
    путем решения f(s*, 0, v*) = 0 методом бисекции.
    Ищем корень функции h(v) = (v/v0)^delta + ( (s0 + T*v) / (s* - l) )^2 - 1 = 0.
    (Это следует из f(s*, 0, v*) = 0 и params['s1_gap_param'] = 0)
    """
    if s_star <= params['l_vehicle_length'] + params['s0_jam_distance']:
        if s_star < params['l_vehicle_length']: # Физически невозможно
            return float('nan') 
        return 0.0
    
    v0_target = params['v0_desired_speed']
    if v0_target <= 0: # Невозможно двигаться вперед
        return 0.0

    # Функция, корень которой мы ищем
    def h_v_star(v_cand):
        if v_cand < 0: return float('inf') 
        # if v_cand > v0_target: return float('inf') # Снято ограничение, т.к. v* может быть > v0 в некоторых моделях

        net_clearance = s_star - params['l_vehicle_length']
        # net_clearance должен быть > 0, т.к. s_star > l + s0
        # Также net_clearance должен быть > s0 для v* > 0

        # term_v_ratio = (v_cand / v0_target)
        # term1 = term_v_ratio**params['delta_accel_exponent']
        # Использование max(0, v_cand) для предотвращения отрицательных оснований степени
        term1 = (max(0, v_cand) / v0_target)**params['delta_accel_exponent']

        s_hat_at_eq = params['s0_jam_distance'] + params['T_safe_time_headway'] * v_cand
        
        if net_clearance <= 1e-9: # Должно быть отфильтровано ранее, но для безопасности
             return float('inf') if s_hat_at_eq > 0 else (1.0 if s_hat_at_eq == 0 else -1.0)


        term2 = (s_hat_at_eq / net_clearance)**2
        return term1 + term2 - 1

    low_v = 0.0
    # Верхняя граница может быть больше v0, если, например, s0 или T отрицательны (нефизично)
    # Для стандартных параметров v* <= v0.
    # Возьмем немного больше v0 на всякий случай.
    high_v = v0_target * 1.1 


    h_low = h_v_star(low_v)
    if abs(h_low) < tol:
        return low_v
    
    h_high = h_v_star(high_v)
    # Если h_high все еще отрицателен, значит корень может быть еще выше
    # Это может произойти если (s0+T*high_v)/(s_star-l) все еще мал.
    # Например, если s_star очень большое.
    if abs(h_high) < tol and h_high == 0 : # h_high может быть 0, если ((s0+Tv0)/(s_star-l))^2 -> 0
         return high_v
    
    # Если h(0) > 0 ( (s0/(s*-l))^2 -1 > 0 => s0 > s*-l => s*-l < s0), то v*=0
    if h_low > 0 :
        # Это условие: (params['s0_jam_distance'] / (s_star - params['l_vehicle_length']))**2 - 1 > 0
        # (s_star - params['l_vehicle_length']) < params['s0_jam_distance']
        # Это соответствует условию s_star < l + s0, которое должно возвращать 0.0 в начале функции.
        # Если мы здесь, значит s_star > l + s0, тогда h_low должно быть < 0.
        # Если нет, то что-то с логикой или параметрами.
        # print(f"Предупреждение find_eq_v: h_low={h_low} > 0 для s*={s_star}. Возврат 0.0")
        return 0.0


    if h_low * h_high > 0:
        # print(f"Предупреждение: Не удается найти равновесную скорость v* для s*={s_star} в диапазоне [{low_v}, {high_v}].")
        # print(f"h({low_v}) = {h_low}, h({high_v}) = {h_high}")
        # Попробуем расширить диапазон high_v, если h_high все еще отрицательный.
        # Это значит, что 1 член еще не перевесил второй.
        if h_high < 0: # (v/v0)^d + ((s0+Tv)/(s*-l))^2 - 1 < 0
                       # Это указывает, что v* > high_v
            # print(f"h_high < 0, пытаемся найти v* > {high_v}")
            # В этом случае решение может быть > v0_target.
            # Это происходит, когда (s* - l) настолько велико, что второй член стремится к 0.
            # Тогда (v/v0)^delta ~ 1, то есть v ~ v0.
            # Если v0_target - целевая скорость, то v* не должно ее сильно превышать.
            # Если h(v0_target) все еще < 0, значит, возможно, v* = v0_target и второй член <0, что невозможно.
            # Скорее всего, означает, что v_star очень близко к v0_target.
            # Либо ошибка в логике, либо параметры таковы, что равновесия нет или оно за пределами v0.
             if abs(h_v_star(params['v0_desired_speed'])) < tol : return params['v0_desired_speed']

        return float('nan') # Не удалось найти интервал для бисекции

    for i in range(max_iter):
        mid_v = (low_v + high_v) / 2
        if mid_v == low_v or mid_v == high_v: # Предотвращение зацикливания при недостаточной точности float
            break
        h_mid = h_v_star(mid_v)

        if abs(h_mid) < tol or (high_v - low_v) / 2 < tol:
            return mid_v

        if h_low * h_mid < 0: # Корень между low_v и mid_v
            high_v = mid_v
            # h_high = h_mid # Не нужно, т.к. h_low не меняется
        else: # Корень между mid_v и high_v
            low_v = mid_v
            h_low = h_mid 
            
    # print(f"Предупреждение: Метод бисекции не сошелся для s*={s_star} за {max_iter} итераций. Последнее значение v*={mid_v:.4f}")
    return mid_v

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

    if v0_target <= 0: # Основные параметры не позволяют движение
        return 0.0, 0.0, 0.0 # или NaN
    
    net_clearance = s_star - l_veh
    if net_clearance <= 1e-3:
        # print(f"Предупреждение: Чистый зазор {net_clearance} слишком мал для расчета производных при s*={s_star}, v*={v_star}.")
        # Если v_star > 0, это не должно происходить, т.к. s_star > l+s0.
        # Если v_star = 0, то s_star = l+s0. net_clearance = s0.
        if v_star == 0 and abs(net_clearance - s0) < 1e-3 : # Это ожидаемый случай при v*=0
             pass # Продолжаем, net_clearance = s0
        elif v_star > 0 : # Это проблема
             return float('nan'), float('nan'), float('nan')
        # Если net_clearance очень мал и v_star=0 (например s_star = l + epsilon)
        # то производные могут быть очень большими.
        # Для простоты, если net_clearance все же слишком мал, возвращаем NaN.
        if net_clearance <= 1e-3: # Повторная проверка после возможного прохода выше
            return float('nan'), float('nan'), float('nan')


    s_hat_eq = s0 + T * v_star

    # f_s = 2 * a_max * s_hat_eq^2 / (net_clearance)^3
    try:
        f_s = 2 * a_max * (s_hat_eq**2) / (net_clearance**3)
    except ZeroDivisionError:
        f_s = float('inf') if a_max * (s_hat_eq**2) > 0 else (float('-inf') if a_max * (s_hat_eq**2) < 0 else 0)


    # f_dv = a_max * s_hat_eq * v_star / ( (net_clearance)^2 * sqrt(a_max * b_decel) )
    sqrt_ab_val = a_max * b_decel
    if sqrt_ab_val <= 0 or v_star < 0: # v_star не должен быть < 0
        f_dv = 0.0 # Если нет возможности тормозить/ускоряться, или скорость отрицательная (нефизично)
    else:
        denominator_f_dv = (net_clearance**2) * math.sqrt(sqrt_ab_val)
        if abs(denominator_f_dv) < 1e-9 : 
            f_dv = float('inf') if a_max * s_hat_eq * v_star > 0 else 0.0 # или другое значение, если числитель 0
        else:
            f_dv = a_max * s_hat_eq * v_star / denominator_f_dv


    # f_v = a_max * [ -delta/v0_target * (v_star/v0_target)^(delta-1) - 2 * s_hat_eq / (net_clearance)^2 * T ]
    # term_v_ratio_deriv calculation
    if v_star == 0:
        if delta == 1: term_v_ratio_deriv = -delta / v0_target
        elif delta < 1: term_v_ratio_deriv = -float('inf') # Проблема для delta < 1
        else: term_v_ratio_deriv = 0.0 # Для delta > 1
    elif v_star < 0 : # Нефизично
        term_v_ratio_deriv = float('nan')
    else: # v_star > 0
        try:
            term_v_ratio_deriv = -delta / v0_target * (v_star / v0_target)**(delta - 1)
        except ZeroDivisionError: # v0_target = 0, уже обработано выше
            term_v_ratio_deriv = float('nan')


    try:
        term_spacing_deriv = -2 * s_hat_eq / (net_clearance**2) * T 
    except ZeroDivisionError:
        term_spacing_deriv = float('-inf') if -2 * s_hat_eq * T > 0 else (float('inf') if -2 * s_hat_eq * T < 0 else 0)

    if math.isnan(term_v_ratio_deriv) or math.isnan(term_spacing_deriv):
        f_v = float('nan')
    elif math.isinf(term_v_ratio_deriv) or math.isinf(term_spacing_deriv):
        # Обработка бесконечностей: если знаки одинаковые, складываем, иначе nan
        if math.isinf(term_v_ratio_deriv) and math.isinf(term_spacing_deriv) and \
           (term_v_ratio_deriv > 0) != (term_spacing_deriv > 0) : # inf - inf
            f_v = float('nan')
        else:
            f_v = a_max * (term_v_ratio_deriv + term_spacing_deriv)
    else:
        f_v = a_max * (term_v_ratio_deriv + term_spacing_deriv)
    
    return f_s, f_dv, f_v

def check_rational_driving_constraints(f_s, f_dv, f_v):
    """Проверяет условия рационального вождения (Ур. 10)."""
    valid_fs = f_s > 1e-9 # Строго больше нуля
    valid_fdv = f_dv >= -1e-9 # Больше или равно нулю (может быть 0, если v_star = 0)
    valid_fv = f_v < -1e-9  # Строго меньше нуля
    
    print("--- Проверка условий рационального вождения ---")
    print(f"f_s = {f_s:.4f} (> 0): {valid_fs}")
    print(f"f_Δv = {f_dv:.4f} (>= 0): {valid_fdv}") 
    print(f"f_v = {f_v:.4f} (< 0): {valid_fv}")
    
    return valid_fs and valid_fdv and valid_fv

def analyze_platoon_stability(f_s, f_dv, f_v):
    """Анализирует устойчивость взвода (Ур. 16)."""
    print("--- Анализ устойчивости взвода (Platoon Stability) ---")
    if any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]):
        print("Невозможно проанализировать: одна из производных NaN или Inf.")
        return False

    coeff_b = f_dv - f_v
    coeff_c = f_s
    
    is_stable = False
    # Условия Рауса-Гурвица: coeff_b > 0 и coeff_c > 0 для устойчивости
    if coeff_b > 1e-9 and coeff_c > 1e-9:
        is_stable = True
        print(f"Коэффициенты характ. уравнения: (f_Δv - f_v) = {coeff_b:.4f}, f_s = {coeff_c:.4f}")
        print("Условия Рауса-Гурвица (все коэффициенты > 0) выполнены.")
        print("Результат: Взвод УСТОЙЧИВ.")
    else:
        print(f"Коэффициенты характ. уравнения: (f_Δv - f_v) = {coeff_b:.4f}, f_s = {coeff_c:.4f}")
        print("Условия Рауса-Гурвица не выполнены или на грани. Анализ корней:")
        try:
            discriminant = coeff_b**2 - 4 * coeff_c
            if discriminant >= 0:
                lambda1 = (-coeff_b + math.sqrt(discriminant)) / 2
                lambda2 = (-coeff_b - math.sqrt(discriminant)) / 2
                print(f"Корни (действительные): λ1 = {lambda1:.4f}, λ2 = {lambda2:.4f}")
                if lambda1 < -1e-9 and lambda2 < -1e-9: # Строго < 0
                    is_stable = True
            else: # Комплексные корни
                real_part = -coeff_b / 2
                print(f"Корни (комплексные): Re(λ) = {real_part:.4f}")
                if real_part < -1e-9: # Строго < 0
                    is_stable = True
            
            if is_stable:
                print("Результат: Взвод УСТОЙЧИВ (на основе анализа корней).")
            else:
                print("Результат: Взвод НЕУСТОЙЧИВ.")
        except OverflowError:
            print("Ошибка OverflowError при расчете дискриминанта/корней.")
            is_stable = False # Не можем определить
            
    return is_stable

def analyze_string_stability(f_s, f_dv, f_v):
    """Анализирует устойчивость цепочки (Ур. 20)."""
    print("--- Анализ устойчивости цепочки (String Stability) ---")
    if any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]):
        print("Невозможно проанализировать: одна из производных NaN или Inf.")
        return False # Неустойчива или неопределенно

    if abs(f_v) < 1e-9:
        print("f_v близко к нулю, анализ устойчивости цепочки может быть неточным / модель вырождена.")
        return False 

    # K = (f_v^2 / 2 - f_Δv*f_v - f_s)
    # Устойчивость цепочки: K > 0 (при f_s > 0, f_v < 0, тогда λ₂ < 0)
    try:
        K_condition = (f_v**2) / 2 - f_dv * f_v - f_s
        print(f"Критерий K = f_v^2/2 - f_Δv*f_v - f_s: {K_condition:.4f}")

        is_stable = False
        # Условия рационального вождения: f_s > 0, f_v < 0 (f_dv >= 0)
        if f_s > 1e-9 and f_v < -1e-9: 
            if K_condition > 1e-9: # K > 0
                is_stable = True 
                print("Результат: Цепочка УСТОЙЧИВА (K > 0).")
            else: # K <= 0
                is_stable = False 
                print("Результат: Цепочка НЕУСТОЙЧИВА (K <= 0).")
        else:
            print("Предупреждение: Условия рационального вождения (f_s > 0, f_v < 0) не выполнены.")
            # Если f_s/f_v^3 > 0 (например, f_s > 0 и f_v > 0, что нерационально)
            # то для устойчивости (λ₂ < 0) нужно K < 0.
            # Но мы придерживаемся случая рационального вождения.
            # Если они не выполнены, поведение λ₂ сложнее интерпретировать просто по знаку K.
            lambda2_numerator = f_s * K_condition
            lambda2_denominator = f_v**3
            if abs(lambda2_denominator) < 1e-9:
                 print("Знаменатель f_v^3 близок к нулю. λ₂ не определен.")
                 is_stable = False
            else:
                lambda2_val = lambda2_numerator / lambda2_denominator
                print(f"Расчетное значение λ₂ ~ {lambda2_val:.4e} (требует осторожной интерпретации).")
                if lambda2_val < -1e-9 :
                    # is_stable = True # Формально да, но ситуация нештатная
                    print("Формально λ₂ < 0, но условия рационального вождения нарушены.")
                else:
                    # is_stable = False
                    print("Формально λ₂ >= 0, и условия рационального вождения нарушены.")
            is_stable = False # Считаем неустойчивой или неопределенной, если рациональность нарушена

    except (ZeroDivisionError, OverflowError) as e:
        print(f"Ошибка при вычислении K или λ₂: {e}")
        is_stable = False 
        
    return is_stable

def collect_data_for_plots(s_star_range, params):
    s_values = []
    v_values = []
    fs_values = []
    fdv_values = []
    fv_values = []
    k_values = [] # K = (f_v^2 / 2 - f_dv*f_v - f_s)
    string_stable_flags = []
    platoon_stable_flags = []

    for s_star in s_star_range:
        v_star = find_equilibrium_velocity(s_star, params)
        
        if v_star is None or math.isnan(v_star):
            # print(f"Пропуск s*={s_star} для графиков: не удалось найти v*")
            continue

        f_s, f_dv, f_v = calculate_partial_derivatives(s_star, v_star, params)

        if any(math.isnan(val) or math.isinf(val) for val in [f_s, f_dv, f_v]):
            # print(f"Пропуск s*={s_star}, v*={v_star} для графиков: NaN/Inf в производных")
            continue
        
        s_values.append(s_star)
        v_values.append(v_star)
        fs_values.append(f_s)
        fdv_values.append(f_dv)
        fv_values.append(f_v)
        
        K_cond = (f_v**2) / 2 - f_dv * f_v - f_s
        k_values.append(K_cond)
        
        # Проверка устойчивости для флагов (упрощенная, без вывода текста)
        # Рациональное вождение
        rational = (f_s > 1e-9 and f_dv >= -1e-9 and f_v < -1e-9)
        
        # Устойчивость взвода
        coeff_b_platoon = f_dv - f_v
        platoon_stable = False
        if rational and coeff_b_platoon > 1e-9 and f_s > 1e-9:
            platoon_stable = True
        platoon_stable_flags.append(platoon_stable)

        # Устойчивость цепочки
        string_stable = False
        if rational and K_cond > 1e-9:
            string_stable = True
        string_stable_flags.append(string_stable)
        
    return {
        's_star': np.array(s_values),
        'v_star': np.array(v_values),
        'f_s': np.array(fs_values),
        'f_dv': np.array(fdv_values),
        'f_v': np.array(fv_values),
        'K_condition': np.array(k_values),
        'platoon_stable': np.array(platoon_stable_flags),
        'string_stable': np.array(string_stable_flags)
    }

def plot_stability_analysis(data, params):
    if len(data['s_star']) == 0:
        print("Нет данных для построения графиков.")
        return

    v_star_kmh = data['v_star'] * 3.6
    # --- График 1: Фундаментальная диаграмма (v* от s*) ---
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(data['s_star'], v_star_kmh, marker='.', linestyle='-', color='b')
    plt.xlabel('Равновесная дистанция s* (м)')
    plt.ylabel('Равновесная скорость v* (км/ч)')
    plt.title('Фундаментальная диаграмма (v* от s*)')
    plt.grid(True)

    # --- График 2: Критерий K от v* ---
    plt.subplot(2, 2, 2)
    # Сортировка по v_star для корректного отображения линии
    sorted_indices = np.argsort(data['v_star'])
    sorted_v_star_kmh = v_star_kmh[sorted_indices]
    sorted_K_condition = data['K_condition'][sorted_indices]
    
    plt.plot(sorted_v_star_kmh, sorted_K_condition, marker='.', linestyle='-', color='r')
    plt.axhline(0, color='black', lw=0.8, linestyle='--')
    plt.xlabel('Равновесная скорость v* (км/ч)')
    plt.ylabel('Критерий K = f_v^2/2 - f_Δv*f_v - f_s')
    plt.title('Критерий устойчивости цепочки K от v*')
    plt.annotate('K > 0 (Цепочка устойчива)', xy=(max(0,np.percentile(sorted_v_star_kmh,20) if len(sorted_v_star_kmh)>0 else 0) , max(0.1, np.percentile(sorted_K_condition, 75) if len(sorted_K_condition)>0 else 0.1)), color='green')
    plt.annotate('K < 0 (Цепочка неустойчива)', xy=(max(0,np.percentile(sorted_v_star_kmh,20) if len(sorted_v_star_kmh)>0 else 0), min(-0.1, np.percentile(sorted_K_condition, 25) if len(sorted_K_condition)>0 else -0.1)), color='red')
    plt.grid(True)

    # --- График 3: Частные производные от v* ---
    plt.subplot(2, 2, 3)
    plt.plot(sorted_v_star_kmh, data['f_s'][sorted_indices], label='f_s (∂f/∂s)', marker='.')
    plt.plot(sorted_v_star_kmh, data['f_dv'][sorted_indices], label='f_Δv (∂f/∂Δv)', marker='.')
    plt.plot(sorted_v_star_kmh, data['f_v'][sorted_indices], label='f_v (∂f/∂v)', marker='.')
    plt.xlabel('Равновесная скорость v* (км/ч)')
    plt.ylabel('Значения производных')
    plt.title('Частные производные от v*')
    plt.legend()
    plt.grid(True)
    # Ограничение по оси Y для f_s, так как она может быть очень большой при малых зазорах
    fs_median = np.median(data['f_s'][sorted_indices]) if len(data['f_s']) > 0 else 1
    fs_std = np.std(data['f_s'][sorted_indices]) if len(data['f_s']) > 0 else 1
    # plt.ylim(bottom=min(np.min(data['f_dv'][sorted_indices]) if len(data['f_dv']) > 0 else -1, np.min(data['f_v'][sorted_indices]) if len(data['f_v']) > 0 else -1, -1), 
    #          top=max(1, fs_median + 3*fs_std))


    # --- График 4: Области устойчивости от v* ---
    plt.subplot(2, 2, 4)
    stable_v_star = v_star_kmh[data['string_stable']]
    unstable_v_star = v_star_kmh[~data['string_stable']]
    
    # Для корректного отображения областей, если v_star не монотонно возрастает с s_star
    # (обычно возрастает, но для общности)
    # Мы уже используем sorted_v_star_kmh
    
    # Покажем просто точки стабильности/нестабильности
    # Более сложная заливка областей требует дополнительной логики определения границ
    
    # Устойчивость взвода
    platoon_stable_points = sorted_v_star_kmh[data['platoon_stable'][sorted_indices]]
    platoon_unstable_points = sorted_v_star_kmh[~data['platoon_stable'][sorted_indices]]
    
    if len(platoon_stable_points) > 0:
        plt.scatter(platoon_stable_points, np.ones(len(platoon_stable_points)) * 1.0, color='cyan', marker='o', label='Взвод Устойчив', alpha=0.7)
    if len(platoon_unstable_points) > 0:
         plt.scatter(platoon_unstable_points, np.ones(len(platoon_unstable_points)) * 1.0, color='magenta', marker='x', label='Взвод Неустойчив', alpha=0.7)


    # Устойчивость цепочки
    string_stable_points = sorted_v_star_kmh[data['string_stable'][sorted_indices]]
    string_unstable_points = sorted_v_star_kmh[~data['string_stable'][sorted_indices]]

    if len(string_stable_points) > 0:
        plt.scatter(string_stable_points, np.ones(len(string_stable_points)) * 0.5, color='green', marker='o', label='Цепочка Устойчива (K>0)', alpha=0.7)
    if len(string_unstable_points) > 0:
        plt.scatter(string_unstable_points, np.ones(len(string_unstable_points)) * 0.5, color='red', marker='x', label='Цепочка Неустойчива (K<=0)', alpha=0.7)

    plt.yticks([0.5, 1.0], ['Уст. Цепочки', 'Уст. Взвода'])
    plt.xlabel('Равновесная скорость v* (км/ч)')
    plt.title('Области устойчивости')
    plt.ylim(0, 1.5)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)

    plt.suptitle(f"Анализ устойчивости IDM (a={params['a_max_accel']:.2f}, T={params['T_safe_time_headway']:.2f})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    params = DEFAULT_IDM_PARAMS.copy()
    # Можно изменить параметры здесь, например:
    # params['a_max_accel'] = 1.0 
    # params['T_safe_time_headway'] = 1.0
    
    # --- Пример анализа для одной точки (как было) ---
    s_star_example = 30.0  # метры
    print(f"--- Анализ для ОДНОЙ точки: s* = {s_star_example} м ---")
    print(f"Используемые параметры IDM: {params}")

    v_star = find_equilibrium_velocity(s_star_example, params)
    if v_star is None or math.isnan(v_star):
        print(f"Не удалось найти равновесную скорость для s* = {s_star_example}")
    else:
        print(f"Равновесная скорость v* = {v_star:.2f} м/с ({(v_star * 3.6):.2f} км/ч)")
        f_s, f_dv, f_v = calculate_partial_derivatives(s_star_example, v_star, params)
        
        if any(math.isnan(val) for val in [f_s, f_dv, f_v]):
            print("Одна или несколько частных производных не определены. Анализ прерван.")
        else:
            print(f"Частные производные в точке равновесия:")
            print(f"  f_s   (∂f/∂s)  = {f_s:.4f}")
            print(f"  f_Δv  (∂f/∂Δv) = {f_dv:.4f}")
            print(f"  f_v   (∂f/∂v)  = {f_v:.4f}")

            rational_ok = check_rational_driving_constraints(f_s, f_dv, f_v)
            if rational_ok:
                analyze_platoon_stability(f_s, f_dv, f_v)
                analyze_string_stability(f_s, f_dv, f_v)
            else:
                print("Анализ устойчивости не проводится из-за нарушения условий рационального вождения.")
    print("-" * 50)

    # --- Сбор данных и построение графиков ---
    print("\n--- Сбор данных для построения графиков ---")
    # Диапазон s_star: от (длина + заторная дистанция + немного) до большого значения
    min_s_star = params['l_vehicle_length'] + params['s0_jam_distance'] + 0.1
    max_s_star = 150 # метры, можно подобрать
    s_star_values_for_plot = np.linspace(min_s_star, max_s_star, 100)
    
    plot_data = collect_data_for_plots(s_star_values_for_plot, params)
    
    if plot_data and len(plot_data['s_star']) > 0 :
        print(f"Собрано {len(plot_data['s_star'])} точек для графиков.")
        plot_stability_analysis(plot_data, params)
    else:
        print("Не удалось собрать данные для графиков.")

    print("\nАнализ завершен.") 