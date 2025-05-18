import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq # Для поиска корней
import argparse # Добавлено для аргументов командной строки
import os # Добавлено для работы с путями
import subprocess # Добавлено для запуска внешних скриптов
from datetime import datetime # Для генерации уникальных имен директорий
import json # Добавлено для чтения JSON
import cmath # Добавлено для комплексных чисел в lambda_max
import pandas as pd # Добавлено для DataFrame и CSV
from scipy.signal import correlate as scipy_correlate, find_peaks # Добавлено для кросс-корреляции и поиска пиков
import csv # Добавлено для сохранения в CSV

"""
Скрипт для анализа линейной устойчивости Интеллектуальной Модели Водителя (IDM).

Этот скрипт реализует методы для:
1.  Расчета параметров IDM и равновесных состояний (скорость, дистанция, поток).
2.  Вычисления частных производных функции ускорения IDM ($f_s, f_{\\Delta v}, f_v$)
    в точках равновесия.
3.  Анализа устойчивости взвода (platoon stability / local stability) на основе
    критериев Рауса-Гурвица.
4.  Анализа устойчивости потока/цепочки (string stability / asymptotic stability)
    двумя методами:
    a.  С использованием критерия K (связанного с $\\lambda_2$ из статьи R.E. Wilson, 2008).
    b.  Путем численного нахождения максимальной действительной части собственных значений
        $\\lambda(k)$ характеристического уравнения для различных волновых чисел $k$.
5.  Построения графиков, иллюстрирующих фундаментальные диаграммы, значения производных,
    критерий K и области устойчивости в зависимости от параметров модели или
    равновесных состояний (чистого зазора $s^*_{net}$ или скорости $v^*$).
6.  Проведения параметрического анализа влияния отдельных параметров IDM (например,
    времени реакции T, максимального ускорения 'a') на устойчивость при
    фиксированных условиях (например, $s^*_{net}$, $v^*$ или поток $Q$).
7.  Опционального запуска симуляций в SUMO с использованием `run_circle_simulation.py`
    для валидации теоретических предсказаний устойчивости и анализа результатов
    симуляций с помощью `analyze_circle_data.py`.

Теоретической основой служат концепции, изложенные в статье
"Car-following models: fifty years of linear stability analysis a mathematical perspective"
(R.E. Wilson and J.A. Wardb, 2010), а также общие принципы анализа IDM.
Подробное описание моделей, формул и методов анализа представлено в файле
THEORETICAL_BACKGROUND.md.

Основные функции:
- `calculate_idm_acceleration`: Расчет ускорения по модели IDM.
- `find_equilibrium_velocity`: Поиск равновесной скорости по чистому зазору.
- `calculate_s_star_for_fixed_v_star`: Поиск равновесного чистого зазора по скорости.
- `find_all_equilibrium_states_for_flow`: Поиск всех равновесных состояний (v, s_net) для потока Q.
- `calculate_partial_derivatives`: Расчет производных $f_s, f_{\\Delta v}, f_v$.
- `analyze_platoon_stability`: Анализ устойчивости взвода.
- `analyze_string_stability`: Анализ устойчивости потока по критерию K.
- `calculate_theoretical_lambda_max`: Анализ устойчивости потока по $\\max Re(\\lambda(k))$.
- `collect_data_for_plots`: Сбор данных для графиков зависимости от $s^*_{net}$.
- `plot_stability_analysis`: Построение этих графиков.
- `collect_data_for_param_sweep`: Сбор данных для параметрического анализа.
- `plot_stability_for_parameter_sweep`: Построение графиков параметрического анализа.
- `main`: Главная функция, управление выполнением, парсинг аргументов командной строки.

Аргументы командной строки:
--sumo-binary: Путь к исполняемому файлу SUMO (sumo-gui.exe или sumo.exe).
--sumo-tools-dir: Путь к директории tools в SUMO.
--run-sumo-simulations: Флаг, указывающий, нужно ли запускать симуляции SUMO.
"""

# >>>>>> ДОБАВЛЕНО ОПРЕДЕЛЕНИЕ ФУНКЦИИ >>>>>>
def calculate_theoretical_lambda_max(f_s, f_dv, f_v, s_e_total, num_k_points=100, k_epsilon=1e-6):
    '''
    Вычисляет максимальный теоретический инкремент Re(lambda(k)) (действительную часть
    собственного значения характеристического уравнения линеаризованной системы)
    путем сканирования волновых чисел k для анализа устойчивости потока (string stability).

    Этот метод основан на анализе характеристического уравнения (аналог Ур. 18 в статье Wilson & Ward, 2010):
    $\\lambda^2 + \\lambda [f_{\\Delta v} - f_v - f_{\\Delta v} e^{-i k s_{e,total}}] + [f_s (1 - e^{-i k s_{e,total}})] = 0$
    Устойчивость потока соответствует $\\max_k (Re(\\lambda(k))) < 0$.

    Args:
        f_s (float): Частная производная функции ускорения по дистанции $s_{net}$
                     в точке равновесия ($df/ds_{net}$).
        f_dv (float): Частная производная функции ускорения по относительной скорости $\\Delta v$
                      в точке равновесия ($df/d(\\Delta v)$).
        f_v (float): Частная производная функции ускорения по скорости $v$
                     в точке равновесия ($df/dv$).
        s_e_total (float): Полная равновесная межцентровая дистанция ($s_{e,net} + l_{vehicle}$).
                           Используется для вычисления фазового сдвига $\\phi = k \\cdot s_{e,total}$.
        num_k_points (int, optional): Количество точек для сканирования волнового числа $k$.
                                     Defaults to 100.
        k_epsilon (float, optional): Малое значение, используемое как нижняя граница для $k$
                                     (кроме $k=0$), чтобы избежать сингулярностей, если $s_{e,total}$ мало.
                                     Defaults to 1e-6.

    Returns:
        float: Максимальное значение $Re(\\lambda(k))$ по всем $k$.
               Возвращает `float('nan')`, если расчет не удался (например, из-за NaN/Inf
               в производных или некорректного $s_{e,total}$).
    '''
    if any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]) or math.isnan(s_e_total) or s_e_total < k_epsilon:
        return float('nan')

    max_re_lambda = -float('inf')

    # Диапазон волновых чисел k. k = 2*pi*n/L, где L = N * s_e_total.
    # В непрерывном пределе k может быть любым. 
    # k=0 соответствует однородному возмущению.
    # Максимальное k соответствует минимальной длине волны ~ 2*s_e_total (предел Найквиста), т.е. k_max ~ pi / s_e_total.
    k_max = math.pi / s_e_total if s_e_total > k_epsilon else math.pi
    # k_values = np.linspace(k_epsilon, k_max, num_k_points) # Старый вариант без k=0
    k_values_positive = np.linspace(k_epsilon, k_max, num_k_points -1 if num_k_points > 1 else 1) # Гарантируем хотя бы одну точку, если num_k_points=1
    k_values = np.insert(k_values_positive, 0, 0.0) # Добавляем k=0 в начало
    k_values = np.unique(k_values) # Убираем дубликаты, если k_epsilon=0 (не должно быть)

    for k in k_values:
        # Характеристическое уравнение для lambda(k) (из линеаризации):
        # lambda^2 + lambda * [f_dv - f_v - f_dv * exp(-i*k*s_e_total)] + 
        #          + [f_s * (1 - exp(-i*k*s_e_total))] = 0
        # Здесь используется полная дистанция s_e_total.
        
        phi = k * s_e_total
        exp_term = cmath.exp(-1j * phi) # Используем cmath для комплексных чисел

        coeff_a_lambda_sq = 1.0
        coeff_b_lambda = f_dv - f_v - f_dv * exp_term
        coeff_c_lambda = f_s * (1.0 - exp_term)

        # Решаем квадратное уравнение a*lambda^2 + b*lambda + c = 0
        try:
            delta_in_sqrt = coeff_b_lambda**2 - 4*coeff_a_lambda_sq*coeff_c_lambda
            # Иногда delta_in_sqrt может быть очень маленьким отрицательным числом из-за ошибок округления,
            # даже если математически оно >=0. cmath.sqrt справится, но для ясности:
            # if abs(delta_in_sqrt.imag) < 1e-12 and delta_in_sqrt.real < 0 and abs(delta_in_sqrt.real) < 1e-12 : 
            #     delta_in_sqrt = complex(0, delta_in_sqrt.imag) # Считаем реальную часть нулем
                
            sqrt_delta = cmath.sqrt(delta_in_sqrt)
            lambda1 = (-coeff_b_lambda + sqrt_delta) / (2*coeff_a_lambda_sq)
            lambda2 = (-coeff_b_lambda - sqrt_delta) / (2*coeff_a_lambda_sq)

            re_lambda1 = lambda1.real
            re_lambda2 = lambda2.real

            current_max_re_for_k = -float('inf')
            if not math.isnan(re_lambda1):
                current_max_re_for_k = max(current_max_re_for_k, re_lambda1)
            if not math.isnan(re_lambda2):
                current_max_re_for_k = max(current_max_re_for_k, re_lambda2)
            
            if current_max_re_for_k > max_re_lambda:
                max_re_lambda = current_max_re_for_k
        except (ZeroDivisionError, OverflowError, ValueError) as e_math:
            # print(f"Math error calculating lambdas for k={k}: {e_math}") # Для отладки
            continue # Пропускаем этот k, если есть проблемы

    if max_re_lambda == -float('inf'):
        return float('nan') 

    return max_re_lambda
# <<<<<< КОНЕЦ ДОБАВЛЕННОГО ОПРЕДЕЛЕНИЯ <<<<<<

# >>>>>> ДОБАВЛЕНО ОПРЕДЕЛЕНИЕ ФУНКЦИИ detect_stop_and_go_waves >>>>>>
def detect_stop_and_go_waves(cameras_coords, Ts, V_data, N_data, axes_vx, axes_vt, axes_heatmap, 
                               Q_data=None, rho_data=None, # <--- НОВЫЕ НЕОБЯЗАТЕЛЬНЫЕ АРГУМЕНТЫ
                               output_csv_path="stop_and_go_events.csv", 
                               CORR_THRESH=0.5, MOVING_AVG_WINDOW=5, 
                               MIN_SPEED_DROP_FACTOR=0.5, MIN_SPEED_FOR_DROP_DETECTION_FACTOR=0.7):
    """
    Анализирует данные с симулированных дорожных камер (детекторов) для обнаружения 
    stop-and-go волн и отмечает их на предоставленных графиках.

    Метод основан на вычислении взаимной корреляции сигналов скорости от пар
    соседних детекторов и последующем определении характеристик волны.

    **Процесс работы:**
    1.  **Подготовка данных:**
        *   Проверяются входные данные (`V_data`, `N_data`, `cameras_coords`).
        *   Рассчитываются вспомогательные характеристики потока: интенсивность (`Q_data`)
            и плотность (`rho_data`), если они не предоставлены как входные аргументы. 
            Эти данные напрямую не используются в текущем алгоритме детектирования волн, 
            но могут быть полезны для отладки или будущих расширений.

    2.  **Итерация по парам камер:** Для каждой пары соседних камер (i, i+1):
        *   **Фильтрация:** Временные ряды средних скоростей с обеих камер 
            (`V_data[i,:]`, `V_data[i+1,:]`) сглаживаются с помощью скользящего 
            среднего (`MOVING_AVG_WINDOW`).
        *   **Взаимная корреляция:** Между отфильтрованными и центрированными 
            (вычтено среднее) сигналами скоростей вычисляется нормированная 
            взаимная корреляция `R(τ)`. Рассматриваются только положительные 
            временные сдвиги `τ > 0`.
        *   **Поиск пика корреляции:** Ищется максимальный пик `R_max` в `R(τ)`, 
            превышающий порог `CORR_THRESH`. Если такой пик найден, 
            соответствующий ему временной лаг `τ_max` (в шагах дискретизации) 
            и значение `R_max` сохраняются.
        *   **Расчет скорости волны (c):** 
            `c = -(x_{i+1} - x_i) / (τ_max * Ts)`, где `x` - координаты камер, 
            `Ts` - период дискретизации. Отрицательный знак указывает на волны, 
            распространяющиеся против движения потока.
        *   **Определение момента начала волны (t_event, x_event):**
            *   `x_event`: Принимается равной координате первой камеры в паре (`cameras_coords[i]`).
            *   `t_event`: Момент времени, когда на первой камере (`i`) происходит 
                "резкое" падение скорости. Падение считается резким, если скорость, 
                будучи выше `MIN_SPEED_FOR_DROP_DETECTION_FACTOR * mean_speed_on_cam_i`, 
                падает ниже `MIN_SPEED_DROP_FACTOR * mean_speed_on_cam_i`. 
                Берется первое такое событие на камере `i`.

    3.  **Формирование списка событий:** Каждое обнаруженное событие (волна) 
        сохраняется как словарь со следующими ключами (см. "Содержимое CSV файла").

    4.  **Аннотация графиков:**
        *   `axes_vx` (график v(x)): Вертикальная пунктирная линия фиолетового цвета 
            в точке `x_event` для каждого обнаруженного фронта волны.
        *   `axes_vt` (график v(t), обычно для первой камеры): Фиолетовый маркер 'X' 
            в точке `(t_event, V_data[0, k_event_on_cam0])`, если волна 
            инициирована на камере 0 (`cam_idx_1 == 0`).
        *   `axes_heatmap` (пространственно-временная диаграмма V(x,t)):
            *   Белый маркер 'x' в точке `(t_event, x_event)` для обозначения начала волны.
            *   Белая пунктирная линия от `(t_event, x_event)` до 
                `(t_event + dt_sync_s, cameras_coords[cam_idx_2])` для визуализации 
                распространения фронта волны. `dt_sync_s` - это `τ_max * Ts`.

    5.  **Сохранение результатов:** Список событий сохраняется в CSV файл.

    **Содержимое CSV файла (`output_csv_path`):**
    Каждая строка (кроме заголовка) содержит информацию об одной обнаруженной волне:
    *   `x_event_m` (float): Координата начала события волны [м].
    *   `t_event_s` (float): Время начала события волны [с].
    *   `wave_speed_mps` (float): Скорость распространения волны [м/с].
    *   `R_max_corr` (float): Максимальное значение нормированной кросс-корреляции.
    *   `dx_m` (float): Расстояние между камерами в паре [м].
    *   `dt_sync_s` (float): Временной лаг (`τ_max * Ts`) между камерами [с].
    *   `cam_idx_1` (int): Индекс первой камеры в паре.
    *   `cam_idx_2` (int): Индекс второй камеры в паре.

    Args:
        cameras_coords (list or np.array): Список или массив координат камер в метрах (x_i).
        Ts (float): Период дискретизации в секундах.
        V_data (np.array): Матрица numpy (num_cameras, num_timesteps) средних скоростей vbar_i[k] (м/с).
        N_data (np.array): Матрица numpy (num_cameras, num_timesteps) числа проехавших машин N_i[k].
        axes_vx (matplotlib.axes.Axes): Объект Axes для графика v(x) для аннотации.
        axes_vt (matplotlib.axes.Axes): Объект Axes для графика v(t) (обычно для первой камеры) для аннотации.
        axes_heatmap (matplotlib.axes.Axes): Объект Axes для spatio-temporal heatmap V(x,t) для аннотации.
        Q_data (np.array, optional): Матрица numpy (num_cameras, num_timesteps) интенсивности Q_i[k] (ТС/с).
                                     Если None, рассчитывается из N_data и Ts.
        rho_data (np.array, optional): Матрица numpy (num_cameras, num_timesteps) плотности rho_i[k] (ТС/м).
                                       Если None, рассчитывается из Q_data и V_data.
        output_csv_path (str, optional): Путь для сохранения CSV файла с событиями.
                                         Defaults to "stop_and_go_events.csv".
        CORR_THRESH (float, optional): Порог для нормализованной кросс-корреляции R_max. 
                                       Событие считается волной, если R_max > CORR_THRESH.
                                       Defaults to 0.5.
        MOVING_AVG_WINDOW (int, optional): Размер окна (в шагах дискретизации) для 
                                           скользящего среднего при фильтрации сигналов скорости.
                                           Defaults to 5.
        MIN_SPEED_DROP_FACTOR (float, optional): Фактор для определения "резкого" падения скорости.
                                                Падение фиксируется, если скорость падает ниже 
                                                `MIN_SPEED_DROP_FACTOR * средняя_скорость_на_камере_i`.
                                                Defaults to 0.5.
        MIN_SPEED_FOR_DROP_DETECTION_FACTOR (float, optional): Фактор для определения "высокой" 
                                                               скорости перед падением. Падение 
                                                               детектируется, только если перед ним 
                                                               скорость была выше 
                                                               `MIN_SPEED_FOR_DROP_DETECTION_FACTOR * средняя_скорость_на_камере_i`.
                                                               Defaults to 0.7.

    Returns:
        list: Список словарей, где каждый словарь представляет обнаруженное событие волны.
              Каждый словарь содержит ключи: 'x_event_m', 't_event_s', 'wave_speed_mps', 
              'R_max_corr', 'dx_m', 'dt_sync_s', 'cam_idx_1', 'cam_idx_2'.
              Возвращает пустой список, если событий не найдено или произошла ошибка.
    """
    # 0. Проверка входных данных и инициализация
    if V_data is None or N_data is None or cameras_coords is None or Ts is None:
        print("Ошибка: V_data, N_data, cameras_coords или Ts не предоставлены.")
        return []
    if V_data.shape != N_data.shape:
        print(f"Ошибка: V_data (shape {V_data.shape}) и N_data (shape {N_data.shape}) должны иметь одинаковый размер.")
        return []
    if V_data.shape[0] != len(cameras_coords):
        print(f"Ошибка: Количество камер в V_data ({V_data.shape[0]}) не совпадает с len(cameras_coords) ({len(cameras_coords)}).")
        return []
    if V_data.ndim != 2 or V_data.shape[0] == 0 or V_data.shape[1] == 0:
        print(f"Ошибка: V_data должен быть 2D массивом с num_cameras > 0 и num_timesteps > 0. Получено shape: {V_data.shape}")
        return []

    num_cameras, num_timesteps = V_data.shape
    events = []

    # 1. Расчет Q и rho, если они не предоставлены
    if Q_data is None:
        if Ts > 1e-9:
            Q_data_calc = N_data / Ts
        else:
            Q_data_calc = np.zeros_like(N_data, dtype=float)
        print("Q_data не предоставлен, рассчитывается из N_data и Ts.")
    else:
        if Q_data.shape != V_data.shape:
            print(f"Предупреждение: Предоставленный Q_data (shape {Q_data.shape}) имеет неверный размер, ожидался {V_data.shape}. Q будет пересчитан.")
            if Ts > 1e-9: Q_data_calc = N_data / Ts
            else: Q_data_calc = np.zeros_like(N_data, dtype=float)
        else:
            Q_data_calc = Q_data
            print("Используется предоставленный Q_data.")

    if rho_data is None:
        # rho = Q / V. Обработка деления на ноль или околонулевую скорость.
        rho_data_calc = np.zeros_like(Q_data_calc, dtype=float)
        # Где V > epsilon, rho = Q/V. Где V близко к 0, а Q > 0, rho -> inf (ставим nan).
        # Где и V, и Q близки к 0, rho = 0.
        mask_v_nonzero = np.abs(V_data) > 1e-3
        rho_data_calc[mask_v_nonzero] = Q_data_calc[mask_v_nonzero] / V_data[mask_v_nonzero]
        
        mask_v_zero_q_nonzero = (~mask_v_nonzero) & (np.abs(Q_data_calc) > 1e-3)
        rho_data_calc[mask_v_zero_q_nonzero] = np.nan 
        print("rho_data не предоставлен, рассчитывается из Q_data и V_data.")
    else:
        if rho_data.shape != V_data.shape:
            print(f"Предупреждение: Предоставленный rho_data (shape {rho_data.shape}) имеет неверный размер, ожидался {V_data.shape}. Rho будет пересчитан.")
            rho_data_calc = np.zeros_like(Q_data_calc, dtype=float)
            mask_v_nonzero = np.abs(V_data) > 1e-3
            rho_data_calc[mask_v_nonzero] = Q_data_calc[mask_v_nonzero] / V_data[mask_v_nonzero]
            mask_v_zero_q_nonzero = (~mask_v_nonzero) & (np.abs(Q_data_calc) > 1e-3)
            rho_data_calc[mask_v_zero_q_nonzero] = np.nan
        else:
            rho_data_calc = rho_data
            print("Используется предоставленный rho_data.")

    # Переменные Q_data и rho_data далее в функции не используются напрямую для детектирования,
    # но они посчитаны и доступны, если понадобятся.
    # Для удобства можно было бы их просто назвать Q_data, rho_data внутри функции,
    # но _calc подчеркивает, что они могли быть вычислены здесь.

    # ... (остальная часть функции detect_stop_and_go_waves без изменений, 
    # она использует V_data для основного анализа)

    for i in range(num_cameras - 1):
        x_i = cameras_coords[i]
        x_i_plus_1 = cameras_coords[i+1]
        delta_x = x_i_plus_1 - x_i
        if abs(delta_x) < 1e-3:
            continue

        vbar_i_raw = V_data[i, :]
        vbar_i_plus_1_raw = V_data[i+1, :]

        vbar_i_filt = pd.Series(vbar_i_raw).rolling(window=MOVING_AVG_WINDOW, center=True, min_periods=1).mean().to_numpy()
        vbar_i_plus_1_filt = pd.Series(vbar_i_plus_1_raw).rolling(window=MOVING_AVG_WINDOW, center=True, min_periods=1).mean().to_numpy()
        
        valid_indices = ~np.isnan(vbar_i_filt) & ~np.isnan(vbar_i_plus_1_filt)
        if np.sum(valid_indices) < MOVING_AVG_WINDOW * 2: 
            continue
            
        v1 = vbar_i_filt[valid_indices]
        v2 = vbar_i_plus_1_filt[valid_indices]

        v1_centered = v1 - np.mean(v1)
        v2_centered = v2 - np.mean(v2)
        
        correlation = scipy_correlate(v1_centered, v2_centered, mode='full')
        
        norm_factor = np.sqrt(np.sum(v1_centered**2) * np.sum(v2_centered**2))
        if norm_factor < 1e-9: 
            normalized_correlation = np.zeros_like(correlation)
        else:
            normalized_correlation = correlation / norm_factor
        
        len_v = len(v1)
        lags = np.arange(-(len_v - 1), len_v)

        positive_lags_indices = np.where(lags > 0)[0]
        if len(positive_lags_indices) == 0:
            continue

        R_positive_lags = normalized_correlation[positive_lags_indices]
        peaks_indices, properties = find_peaks(R_positive_lags, height=CORR_THRESH)

        if len(peaks_indices) == 0:
            continue

        idx_of_max_peak_in_R_positive = peaks_indices[np.argmax(properties['peak_heights'])]
        R_max_val = properties['peak_heights'][np.argmax(properties['peak_heights'])]
        tau_max_steps = lags[positive_lags_indices[idx_of_max_peak_in_R_positive]]
        
        tau_max_time = tau_max_steps * Ts
        if abs(tau_max_time) < 1e-9:
            wave_speed = float('inf') if delta_x != 0 else float('nan')
        else:
            wave_speed = -delta_x / tau_max_time

        mean_speed_on_cam_i = np.mean(vbar_i_filt[~np.isnan(vbar_i_filt)])
        if np.isnan(mean_speed_on_cam_i): continue

        speed_threshold_low = MIN_SPEED_DROP_FACTOR * mean_speed_on_cam_i
        speed_threshold_high = MIN_SPEED_FOR_DROP_DETECTION_FACTOR * mean_speed_on_cam_i

        potential_k_peaks = []
        was_above_high = False
        for k_step in range(len(vbar_i_filt)):
            if np.isnan(vbar_i_filt[k_step]): continue
            if was_above_high and vbar_i_filt[k_step] < speed_threshold_low:
                potential_k_peaks.append(k_step) 
                was_above_high = False 
            if vbar_i_filt[k_step] > speed_threshold_high:
                was_above_high = True
            elif vbar_i_filt[k_step] < speed_threshold_low :
                was_above_high = False

        if not potential_k_peaks:
            continue
        
        k_peak = potential_k_peaks[0]
        t_event = k_peak * Ts 
        x_event = x_i

        event_data = {
            'x_event_m': x_event,
            't_event_s': t_event,
            'wave_speed_mps': wave_speed,
            'R_max_corr': R_max_val,
            'dx_m': delta_x,
            'dt_sync_s': tau_max_time,
            'cam_idx_1': i,
            'cam_idx_2': i + 1
        }
        events.append(event_data)
        
    added_label_vx = False
    added_label_vt = False
    added_label_heatmap = False

    for event in events:
        xe = event['x_event_m']
        te = event['t_event_s']
        cam_idx_1 = event['cam_idx_1']
        cam_idx_2 = event['cam_idx_2']
        dt_sync_s = event['dt_sync_s']
        
        if axes_vx:
            label_vx = "Stop-and-Go Wave Front (x)" if not added_label_vx else None
            axes_vx.axvline(x=xe, color='purple', linestyle='--', alpha=0.7, label=label_vx)
            added_label_vx = True

        # Аннотируем график v(t) (для камеры 0) только если волна началась на камере 0
        if axes_vt and cam_idx_1 == 0:
            k_event_on_cam0 = int(round(te / Ts))
            if 0 <= k_event_on_cam0 < num_timesteps:
                v_at_cam0_at_te = V_data[0, k_event_on_cam0]
                if not np.isnan(v_at_cam0_at_te):
                    label_vt = "Wave Arrival @ Cam0" if not added_label_vt else None
                    axes_vt.plot(te, v_at_cam0_at_te, marker='X', color='purple', markersize=8, alpha=0.7, label=label_vt)
                    # Обновляем маркер, чтобы следующая легенда для этого типа события не дублировалась
                    # Это необходимо, если несколько волн начинаются на камере 0
                    if label_vt: added_label_vt = True 
        
        if axes_heatmap:
            label_hm_marker = "Wave Start (t,x)" if not added_label_heatmap else None
            axes_heatmap.plot(te, xe, 'wx', markersize=7, markeredgewidth=1.5, alpha=0.9, label=label_hm_marker)
            
            # Рисуем линию распространения волны
            x_end_wave = cameras_coords[cam_idx_2]
            t_end_wave = te + dt_sync_s
            label_hm_line = "Wave Propagation" if not added_label_heatmap and not label_hm_marker else None # Легенда для линии, если еще не было
            axes_heatmap.plot([te, t_end_wave], [xe, x_end_wave], color='white', linestyle='--', linewidth=1.5, alpha=0.8, label=label_hm_line)
            
            if label_hm_marker or label_hm_line:
                 added_label_heatmap = True
            
    if added_label_vx and axes_vx: axes_vx.legend(fontsize='small')
    if added_label_vt and axes_vt: axes_vt.legend(fontsize='small')
    if added_label_heatmap and axes_heatmap: axes_heatmap.legend(fontsize='small')

    if events:
        print(f"\nОбнаружено {len(events)} событий stop-and-go волн:")
        event_df = pd.DataFrame(events)
        print(event_df.to_string())
        try:
            event_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"События сохранены в: {output_csv_path}")
        except Exception as e:
            print(f"Ошибка при сохранении событий в CSV: {e}")
    else:
        print("\nStop-and-go волны не обнаружены.")
        try:
            header = ['x_event_m', 't_event_s', 'wave_speed_mps', 'R_max_corr', 'dx_m', 'dt_sync_s', 'cam_idx_1', 'cam_idx_2']
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            print(f"Пустой файл событий (с заголовками) сохранен в: {output_csv_path}")
        except Exception as e:
            print(f"Ошибка при сохранении пустого CSV файла событий: {e}")

    return events
# <<<<<< КОНЕЦ ДОБАВЛЕННОГО ОПРЕДЕЛЕНИЯ ФУНКЦИИ detect_stop_and_go_waves <<<<<<

# Стандартные параметры IDM из статьи (Таблица 1, s1=0)
DEFAULT_IDM_PARAMS = {
    'v0_desired_speed': 33.3,  # m/s (120 km/h) - Желаемая скорость v0
    'T_safe_time_headway': 1.6, # s - Безопасная временная дистанция T
    'a_max_accel': 2.5,       # m/s^2 (в статье это 'a') - Максимальное ускорение a
    'b_comfort_decel': 4.6,   # m/s^2 (в статье это 'b') - Комфортное замедление b
    'delta_accel_exponent': 4, # безразмерный (в статье это 'delta' или 'd') - Экспонента ускорения delta
    's0_jam_distance': 2.0,    # m (чистая дистанция в заторе) - Минимальный чистый зазор s0
    'l_vehicle_length': 5.0,   # m (длина автомобиля) - Длина автомобиля l
    's1_gap_param': 0.0        # m (параметр s1, обычно 0 для IDM по статье) - Параметр доп. зазора s1
}

def calculate_s_hat(v, dv, params):
    """
    Вычисляет желаемую динамическую дистанцию (чистый зазор) ŝ(v, Δv) согласно IDM.

    Это расстояние, которое водитель стремится поддерживать.
    Формула соответствует Ур. 14 из статьи Wilson & Ward (2010) при s1=0:
    ŝ(v, Δv) = s0 + T*v - (v*Δv) / (2*sqrt(a*b))
    Функция обеспечивает, что возвращаемый зазор не отрицателен.

    Args:
        v (float): Текущая скорость автомобиля (м/с).
        dv (float): Относительная скорость (v_leader - v_follower) (м/с).
        params (dict): Словарь параметров IDM, должен содержать:
                       's0_jam_distance', 'T_safe_time_headway',
                       'a_max_accel', 'b_comfort_decel'.
                       Предполагается, что 's1_gap_param' = 0.

    Returns:
        float: Желаемый чистый зазор ŝ (м). Не может быть отрицательным.
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
    Вычисляет ускорение согласно Интеллектуальной Модели Водителя (IDM).

    Формула соответствует Ур. 13 из статьи Wilson & Ward (2010), где s_net_clearance
    используется вместо (s-l):
    a_IDM = a * [1 - (v/v0)^delta - (ŝ(v,Δv)/s_net_clearance)^2]

    Args:
        s_net_clearance (float): Текущий чистый зазор (м) до впереди идущего автомобиля.
        dv (float): Относительная скорость (v_leader - v_follower) (м/с).
        v (float): Текущая скорость автомобиля (м/с).
        params (dict): Словарь параметров IDM, должен содержать:
                       'a_max_accel', 'v0_desired_speed', 'delta_accel_exponent',
                       а также параметры, необходимые для `calculate_s_hat`.

    Returns:
        float: Ускорение автомобиля (м/с²).
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
    Находит равновесную скорость v* для заданной равновесной чистой дистанции s*_net.

    Равновесная скорость - это скорость, при которой ускорение IDM равно нулю,
    когда относительная скорость Δv = 0, а чистый зазор равен s*_net.
    Решается уравнение: calculate_idm_acceleration(s*_net, 0, v*, params) = 0
    относительно v*. Используется метод Брента для поиска корня.

    Args:
        s_star_net (float): Равновесный чистый зазор (м).
        params (dict): Словарь параметров IDM.
        tol (float, optional): Допустимая погрешность для поиска корня. Defaults to 1e-6.
        max_iter (int, optional): Максимальное количество итераций для метода Брента. Defaults to 100.

    Returns:
        float: Равновесная скорость v* (м/с) или `float('nan')`, если не найдена.
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
    Вычисляет равновесный чистый зазор s*_net для заданной равновесной скорости v*.

    Равновесный чистый зазор - это такой зазор, при котором ускорение IDM равно нулю,
    когда скорость равна fixed_v_star, а относительная скорость Δv = 0.
    Решается уравнение: calculate_idm_acceleration(s*_net, 0, fixed_v_star, params) = 0
    относительно s*_net.

    Args:
        fixed_v_star (float): Заданная равновесная скорость (м/с).
        params (dict): Словарь параметров IDM.
        tol (float, optional): Допуск для сравнения значений. Defaults to 1e-9.

    Returns:
        float: Равновесный чистый зазор s*_net (м) или `float('nan')`, если не найден или нефизичен.
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
    Находит все равновесные состояния (скорость v_e, чистый зазор s_e_net)
    для заданного транспортного потока Q.

    Равновесное состояние удовлетворяет двум условиям:
    1. Ускорение IDM равно нулю: calculate_idm_acceleration(s_e_net, 0, v_e, params) = 0.
    2. Связь потока, скорости и полного зазора: Q = v_e / (s_e_net + l_vehicle).

    Функция сканирует диапазон возможных скоростей и использует метод Брента
    для поиска корней функции, выражающей ускорение через v_e (при условии связи Q, v_e, s_e_net).
    Это позволяет найти несколько равновесных состояний, если они существуют
    (например, для свободной и заторной ветвей фундаментальной диаграммы).

    Args:
        target_Q_veh_per_sec (float): Целевой поток (автомобилей/сек).
        params (dict): Словарь параметров IDM.
        v_search_min_abs (float, optional): Минимальная абсолютная скорость для поиска. Defaults to 1e-3.
        xtol_brentq (float, optional): Точность для метода Брента. Defaults to 1e-6.
        maxiter_brentq (int, optional): Макс. итераций для Брента. Defaults to 100.
        num_scan_intervals (int, optional): Количество интервалов для сканирования диапазона скоростей.
                                          Defaults to 200.
        verbose (bool, optional): Флаг для вывода отладочной информации. Defaults to False.

    Returns:
        list[tuple[float, float]]: Список кортежей (v_e, s_e_net), представляющих
                                   найденные равновесные состояния.
                                   Может быть пустым, если решения не найдены.
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
    Вычисляет частные производные функции ускорения IDM (f_s, f_dv, f_v)
    в точке равновесия (s*_net, Δv=0, v*).

    Производные:
    - f_s = ∂f/∂s_net : чувствительность к изменению чистого зазора.
    - f_dv = ∂f/∂(Δv) : чувствительность к изменению относительной скорости.
    - f_v = ∂f/∂v : чувствительность к изменению собственной скорости.

    Эти производные используются для линейного анализа устойчивости.
    Расчеты основаны на аналитическом дифференцировании уравнений IDM.

    Args:
        s_star_net (float): Равновесный чистый зазор (м).
        v_star (float): Равновесная скорость (м/с).
        params (dict): Словарь параметров IDM.
        tol (float, optional): Допуск для проверок. Defaults to 1e-6.

    Returns:
        tuple[float, float, float]: Кортеж (f_s, f_dv, f_v).
                                    Возвращает (NaN, NaN, NaN) при нефизичных входных данных.
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
    """
    Проверяет выполнение "условий рационального вождения".

    Условия (согласно Ур. 10 в статье Wilson & Ward, 2010, с небольшим допуском для f_dv):
    - f_s > 0: Увеличение зазора ведет к ускорению.
    - f_dv >= 0: Увеличение относительной скорости (лидер удаляется) ведет к ускорению.
    - f_v < 0: Увеличение собственной скорости ведет к замедлению (или меньшему ускорению).

    Args:
        f_s (float): Производная df/ds_net.
        f_dv (float): Производная df/d(Δv).
        f_v (float): Производная df/dv.

    Returns:
        bool: True, если все условия выполнены, иначе False.
    """
    valid_fs = f_s > 1e-9 
    valid_fdv = f_dv >= -1e-9 # Может быть 0, если v_star=0 или T=0 и s0=0 в s_hat
    valid_fv = f_v < -1e-9  
    return valid_fs and valid_fdv and valid_fv

def analyze_platoon_stability(f_s, f_dv, f_v, verbose=True):
    """
    Анализирует устойчивость взвода (platoon/local stability).

    Устойчивость взвода определяется по корням характеристического уравнения
    (Ур. 16 в статье Wilson & Ward, 2010):
    λ_plat^2 + (f_dv - f_v)λ_plat + f_s = 0
    Взвод устойчив, если Re(λ_plat) < 0 для обоих корней.
    Это эквивалентно условиям Рауса-Гурвица: (f_dv - f_v) > 0 и f_s > 0.

    Args:
        f_s (float): Производная df/ds_net.
        f_dv (float): Производная df/d(Δv).
        f_v (float): Производная df/dv.
        verbose (bool, optional): Выводить ли детальную информацию. Defaults to True.

    Returns:
        bool: True, если взвод устойчив, иначе False.
    """
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
    """
    Анализирует устойчивость потока/цепочки (string/asymptotic stability)
    с использованием критерия K.

    Критерий K (связан с λ₂ из Ур. 20 статьи Wilson & Ward, 2010):
    K = f_v²/2 - f_dv*f_v - f_s
    При выполнении условий рационального вождения ($f_s > 0, f_v < 0$),
    поток устойчив, если K > 0.

    Args:
        f_s (float): Производная df/ds_net.
        f_dv (float): Производная df/d(Δv).
        f_v (float): Производная df/dv.
        verbose (bool, optional): Выводить ли детальную информацию. Defaults to True.

    Returns:
        tuple[bool, float]: Кортеж (is_stable, K_value).
                            is_stable: True, если поток устойчив по этому критерию.
                            K_value: Значение критерия K.
    """
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
        string_stable, K = analyze_string_stability(f_s,f_dv,f_v, verbose=False)
        
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
    """
    Строит комплексный график анализа устойчивости IDM на основе данных,
    собранных как функция от равновесного чистого зазора s*_net.

    График состоит из четырех субграфиков (2x2):

    1.  **Верхний левый: Фундаментальная диаграмма (v* от s*_net)**
        -   Ось X: Равновесный чистый зазор $s^*_{net}$ (м).
        -   Ось Y: Равновесная скорость $v^*$ (км/ч).
        -   Отображает, как изменяется равновесная скорость при увеличении
            равновесного чистого зазора между автомобилями.

    2.  **Верхний правый: Критерий K от v***
        -   Ось X: Равновесная скорость $v^*$ (км/ч).
        -   Ось Y: Значение критерия $K = f_v^2/2 - f_{\Delta v}f_v - f_s$.
        -   Критерий K используется для определения устойчивости потока (string stability).
            Поток теоретически устойчив, если $K > 0$ (при выполнении условий
            рационального вождения).
        -   Горизонтальная линия на уровне $K=0$ показывает границу устойчивости.

    3.  **Нижний левый: Частные производные от v***
        -   Ось X: Равновесная скорость $v^*$ (км/ч).
        -   Ось Y: Значения частных производных $f_s$, $f_{\Delta v}$, и $f_v$.
        -   Эти производные характеризуют чувствительность функции ускорения IDM
            к изменениям зазора, относительной скорости и собственной скорости
            соответственно. Они являются основой для линейного анализа устойчивости.

    4.  **Нижний правый: Области устойчивости от v***
        -   Ось X: Равновесная скорость $v^*$ (км/ч).
        -   Ось Y: Дискретные уровни, представляющие устойчивость взвода и потока.
        -   **Устойчивость взвода (Platoon Stability)**:
            -   Синий кружок ('c', 'o'): Взвод устойчив.
            -   Пурпурный крестик ('m', 'x'): Взвод неустойчив.
        -   **Устойчивость потока/цепочки (String Stability, по критерию K)**:
            -   Зеленый кружок ('g', 'o'): Поток устойчив ($K>0$ и рац. вождение).
            -   Красный крестик ('r', 'x'): Поток неустойчив ($K \le 0$ или не рац. вождение).
        -   Этот график наглядно показывает, при каких равновесных скоростях
            различные типы устойчивости выполняются или нарушаются.

    Данные для построения должны быть отсортированы по $v^*$ для корректного
    отображения на графиках 2, 3 и 4.

    Args:
        data (dict): Словарь с данными, полученный от `collect_data_for_plots`.
                     Должен содержать ключи: 's_star_net', 'v_star', 'f_s', 'f_dv',
                     'f_v', 'K_condition', 'platoon_stable', 'string_stable'.
        params (dict): Словарь параметров IDM (используется для общего заголовка графика).
    """
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
    lambda_max_theory_list = [] # Добавлено для lambda_max_theory

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
            print(f"current_s_net_eq={current_s_net_eq:.2f}, current_v_eq={current_v_eq:.2f}, T={param_to_sweep_key}, val={val}")
            f_s, f_dv, f_v = calculate_partial_derivatives(current_s_net_eq, current_v_eq, current_params)
            
            current_s_total_eq = current_s_net_eq + current_params['l_vehicle_length'] # Добавлено для lambda_max_theory
            lambda_theory = calculate_theoretical_lambda_max(f_s, f_dv, f_v, current_s_total_eq) if not any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]) else float('nan') # Добавлено
            
            if any(math.isnan(x) or math.isinf(x) for x in [f_s, f_dv, f_v]):
                if verbose: print(f"Пропуск {param_to_sweep_key}={val} (s_net={current_s_net_eq:.2f}, v={current_v_eq:.2f}): NaN/Inf в производных")
                # Заполняем пустыми значениями, чтобы сохранить соответствие с param_values
                param_vals_list.append(val)
                s_star_net_list.append(current_s_net_eq) # или float('nan') если хотим четко пометить
                v_star_list.append(current_v_eq)       # или float('nan')
                K_list.append(float('nan'))
                platoon_stable_list.append(False) # или None/NaN
                string_stable_list.append(False)  # или None/NaN
                lambda_max_theory_list.append(float('nan')) # Добавлено
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
            lambda_max_theory_list.append(lambda_theory) # Добавлено
        
    return {
        'param_values': np.array(param_vals_list), 
        's_star_net': np.array(s_star_net_list), 
        'v_star': np.array(v_star_list), 
        'K_condition': np.array(K_list),
        'platoon_stable': np.array(platoon_stable_list), 
        'string_stable': np.array(string_stable_list),
        'lambda_max_theory': np.array(lambda_max_theory_list) # Добавлено
    }

    """
    Строит графики результатов параметрического анализа устойчивости IDM,
    показывая влияние одного варьируемого параметра на равновесные состояния и их устойчивость.

    График состоит из трех субграфиков, расположенных вертикально:

    1.  **Верхний график: Зависимость $v^*$ и $s^*_{net}$ от варьируемого параметра**
        -   Ось X: Значения варьируемого параметра (например, время реакции T).
        -   Левая ось Y: Равновесная скорость $v^*$ (км/ч).
        -   Правая ось Y: Равновесный чистый зазор $s^*_{net}$ (м).
        -   Если для одного значения варьируемого параметра существует несколько
            равновесных состояний (например, при фиксированном потоке $Q$, что дает
            нижнюю и верхнюю ветви фундаментальной диаграммы), они отображаются
            разными цветами/маркерами:
            -   **Нижняя ветвь (или единственное состояние):**
                -   $v^*$: красные кружки ('red', 'o').
                -   $s^*_{net}$: темно-красные крестики ('darkred', 'x').
            -   **Верхняя ветвь:**
                -   $v^*$: синие кружки ('blue', 'o').
                -   $s^*_{net}$: темно-синие крестики ('darkblue', 'x').
            -   **Единственное состояние (если ветвей нет):**
                -   $v^*$: черные кружки ('black', 'o').
                -   $s^*_{net}$: темно-серые крестики ('dimgray', 'x').

    2.  **Средний график: Критерий K от варьируемого параметра**
        -   Ось X: Значения варьируемого параметра.
        -   Ось Y: Значение критерия $K = f_v^2/2 - f_{\Delta v}f_v - f_s$.
        -   Отображает $K$ для каждой ветви равновесных состояний:
            -   Нижняя ветвь: красно-оранжевая линия ('salmon').
            -   Верхняя ветвь: светло-синяя линия ('skyblue').
            -   Единственное состояние: серая линия ('gray').
        -   Горизонтальная линия на уровне $K=0$ показывает границу теоретической
            устойчивости потока ($K>0$ означает устойчивость).

    3.  **Нижний график: Устойчивость потока по ветвям от варьируемого параметра**
        -   Ось X: Значения варьируемого параметра.
        -   Ось Y: Два уровня, представляющие нижнюю/единственную и верхнюю ветви.
        -   **Теоретическая устойчивость потока (String Stability по критерию K):**
            -   Отображается закрашенными квадратами ('s') на соответствующем уровне ветви.
            -   Зеленый квадрат: Теоретически устойчиво ($K>0$ и рац. вождение).
            -   Красный квадрат: Теоретически неустойчиво ($K \le 0$ или не рац. вождение).
        -   **Экспериментальная устойчивость (из результатов симуляций SUMO, если предоставлены):**
            -   Отображается большими крестами ('X') поверх теоретических маркеров,
                на соответствующем уровне ветви, к которой относится симуляция.
            -   Ярко-зеленый крест ('lime', edgecolor='darkgreen'): Экспериментально устойчиво
                (волны не наблюдались в симуляции).
            -   Томатный крест ('tomato', edgecolor='darkred'): Экспериментально неустойчиво
                (волны наблюдались в симуляции).
        -   Присутствует легенда, поясняющая маркеры теоретической и экспериментальной устойчивости.

    Args:
        data (dict): Данные, собранные функцией `collect_data_for_param_sweep`.
        swept_param_key (str): Ключ варьируемого параметра IDM (например, 'T_safe_time_headway').
        swept_param_label (str): Метка для оси X (например, "Время реакции T (с)").
        fixed_condition_label (str): Описание фиксированного условия (например, "Q = 1800 авто/час").
        base_params (dict): Базовый набор параметров IDM (используется для общего заголовка графика).
        simulation_results (list[dict], optional): Список словарей с результатами
                                                  симуляций SUMO. Каждый словарь должен
                                                  содержать как минимум ключ варьируемого
                                                  параметра (например, 'T'), 'v_star' (м/с)
                                                  соответствующей теоретической точки равновесия,
                                                  и 'waves_observed' (bool).
                                                  Defaults to None.
    """
def plot_stability_for_parameter_sweep(data, swept_param_key, swept_param_label, fixed_condition_label, base_params, simulation_results=None):
    if len(data['param_values']) == 0:
        print(f"Нет данных для построения графиков для параметра {swept_param_label}.")
        return

    grouped_data = {}
    for i, param_val in enumerate(data['param_values']):
        if param_val not in grouped_data:
            grouped_data[param_val] = []
        # Убедимся, что K_condition и string_stable собираются 
        grouped_data[param_val].append({
            'v_star': data['v_star'][i],
            's_star_net': data['s_star_net'][i],
            'K_condition': data['K_condition'][i], # K используется для определения ss_lower/upper
            'platoon_stable': data['platoon_stable'][i],
            'string_stable': data['string_stable'][i], # Это флаг теор. уст-ти (K>0 при рац. вожд.)
            'lambda_max_theory': data['lambda_max_theory'][i] if 'lambda_max_theory' in data else float('nan') 
        })

    # Собираем данные для каждой ветви, включая K_condition и string_stable
    param_x_lower, v_lower, s_net_lower, k_lower, ps_lower, ss_lower = [], [], [], [], [], []
    param_x_upper, v_upper, s_net_upper, k_upper, ps_upper, ss_upper = [], [], [], [], [], []
    param_x_single, v_single, s_net_single, k_single, ps_single, ss_single = [], [], [], [], [], []
    # lambda_theory_* больше не нужны для этих графиков, но оставим их сбор, если они используются где-то еще
    lambda_theory_lower, lambda_theory_upper, lambda_theory_single = [], [], [] 

    unique_param_values_sorted = sorted(grouped_data.keys())

    for param_val in unique_param_values_sorted:
        states = grouped_data[param_val]
        valid_states = [s for s in states if not math.isnan(s['v_star']) and not math.isnan(s['s_star_net'])]
        
        if not valid_states:
            continue

        if len(valid_states) == 1:
            state = valid_states[0]
            param_x_single.append(param_val)
            v_single.append(state['v_star'])
            s_net_single.append(state['s_star_net'])
            k_single.append(state['K_condition']) 
            ps_single.append(state['platoon_stable'])
            ss_single.append(state['string_stable']) # Собираем string_stable (теор. уст.)
            lambda_theory_single.append(state.get('lambda_max_theory', float('nan')))
        elif len(valid_states) >= 2:
            valid_states.sort(key=lambda x: x['v_star'])
            lower_state = valid_states[0]
            upper_state = valid_states[-1]
            
            param_x_lower.append(param_val)
            v_lower.append(lower_state['v_star'])
            s_net_lower.append(lower_state['s_star_net'])
            k_lower.append(lower_state['K_condition']) 
            ps_lower.append(lower_state['platoon_stable'])
            ss_lower.append(lower_state['string_stable']) # Собираем string_stable (теор. уст.)
            lambda_theory_lower.append(lower_state.get('lambda_max_theory', float('nan')))

            param_x_upper.append(param_val)
            v_upper.append(upper_state['v_star'])
            s_net_upper.append(upper_state['s_star_net'])
            k_upper.append(upper_state['K_condition']) 
            ps_upper.append(upper_state['platoon_stable'])
            ss_upper.append(upper_state['string_stable']) # Собираем string_stable (теор. уст.)
            lambda_theory_upper.append(upper_state.get('lambda_max_theory', float('nan')))
            
            if len(valid_states) > 2:
                 print(f"Предупреждение: найдено {len(valid_states)} состояний для {swept_param_key}={param_val}. Используются только нижнее и верхнее.")

    # Преобразуем в numpy массивы
    param_x_lower, v_lower, s_net_lower, k_lower, ps_lower, ss_lower = np.array(param_x_lower), np.array(v_lower), np.array(s_net_lower), np.array(k_lower), np.array(ps_lower, dtype=bool), np.array(ss_lower, dtype=bool)
    param_x_upper, v_upper, s_net_upper, k_upper, ps_upper, ss_upper = np.array(param_x_upper), np.array(v_upper), np.array(s_net_upper), np.array(k_upper), np.array(ps_upper, dtype=bool), np.array(ss_upper, dtype=bool)
    param_x_single, v_single, s_net_single, k_single, ps_single, ss_single = np.array(param_x_single), np.array(v_single), np.array(s_net_single), np.array(k_single), np.array(ps_single, dtype=bool), np.array(ss_single, dtype=bool)
    # lambda_theory_lower, lambda_theory_upper, lambda_theory_single = np.array(lambda_theory_lower), np.array(lambda_theory_upper), np.array(lambda_theory_single) # Не используются в этих графиках напрямую

    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True) # Изменено на 3 субплота, скорректирован размер

    # --- График axs[0]: v* и s*_net (как и был) ---
    ax0 = axs[0]
    ax0_twin = ax0.twinx()
    handles0, labels0 = [], [] # Собираем handles/labels для объединенной легенды
    if len(param_x_lower) > 0:
        h_v_l = ax0.scatter(param_x_lower, v_lower * 3.6, marker='o', s=25, alpha=0.7, color='red', label='v* (Нижн. ветвь, км/ч)')
        h_s_l = ax0_twin.scatter(param_x_lower, s_net_lower, marker='x', s=25, alpha=0.7, color='darkred', label='s*_net (Нижн. ветвь, м)')
        handles0.extend([h_v_l, h_s_l]); labels0.extend([h_v_l.get_label(), h_s_l.get_label()])
    if len(param_x_upper) > 0:
        h_v_u = ax0.scatter(param_x_upper, v_upper * 3.6, marker='o', s=25, alpha=0.7, color='blue', label='v* (Верхн. ветвь, км/ч)')
        h_s_u = ax0_twin.scatter(param_x_upper, s_net_upper, marker='x', s=25, alpha=0.7, color='darkblue', label='s*_net (Верхн. ветвь, м)')
        handles0.extend([h_v_u, h_s_u]); labels0.extend([h_v_u.get_label(), h_s_u.get_label()])
    if len(param_x_single) > 0:
        h_v_s = ax0.scatter(param_x_single, v_single * 3.6, marker='o', s=25, alpha=0.7, color='black', label='v* (Единств., км/ч)')
        h_s_s = ax0_twin.scatter(param_x_single, s_net_single, marker='x', s=25, alpha=0.7, color='dimgray', label='s*_net (Единств., м)')
        handles0.extend([h_v_s, h_s_s]); labels0.extend([h_v_s.get_label(), h_s_s.get_label()])
    ax0.set_ylabel('Равновесная скорость v* (км/ч)')
    ax0_twin.set_ylabel('Равновесный чистый зазор s*_net (м)')
    if handles0: ax0.legend(handles=handles0, labels=labels0, loc='best', fontsize='small')
    ax0.set_title(f'Зависимость v* и s*_net от {swept_param_label}')
    ax0.grid(True)

    # --- График axs[1]: КРИТЕРИЙ K (остается без изменений) ---
    ax_k_plot = axs[1]
    handles_k, labels_k = [], []
    if len(param_x_lower) > 0 and len(k_lower) == len(param_x_lower):
        h = ax_k_plot.plot(param_x_lower, k_lower, linestyle='-', marker='.', color='salmon', label='K (Нижн. ветвь)')
        handles_k.append(h[0]); labels_k.append(h[0].get_label())
    if len(param_x_upper) > 0 and len(k_upper) == len(param_x_upper):
        h = ax_k_plot.plot(param_x_upper, k_upper, linestyle='-', marker='.', color='skyblue', label='K (Верхн. ветвь)')
        handles_k.append(h[0]); labels_k.append(h[0].get_label())
    if len(param_x_single) > 0 and len(k_single) == len(param_x_single):
        h = ax_k_plot.plot(param_x_single, k_single, linestyle='-', marker='.', color='gray', label='K (Единств.)')
        handles_k.append(h[0]); labels_k.append(h[0].get_label())
    ax_k_plot.axhline(0, color='black', lw=0.8, linestyle='--') # K=0 - важный порог
    ax_k_plot.set_ylabel('Критерий K')
    ax_k_plot.set_title(f'Критерий K от {swept_param_label}')
    if handles_k: ax_k_plot.legend(handles=handles_k, labels=labels_k, fontsize='small', loc='best')
    ax_k_plot.grid(True)

    # --- График axs[2]: УСТОЙЧИВОСТЬ ПО ВЕТВЯМ СКОРОСТЕЙ (ранее axs[3]) ---
    ax_branch_stability = axs[2] # Теперь это axs[2]
    # branch_stability_handles, branch_stability_labels = [], [] # Больше не нужны для сборки сложных ручных легенд
    ax_branch_stability.set_title(f'Устойчивость по ветвям от {swept_param_label}')
    y_lower_branch, y_upper_branch = 0.3, 0.7
    ax_branch_stability.set_yticks([y_lower_branch, y_upper_branch])
    ax_branch_stability.set_yticklabels(['Нижняя/Единств. ветвь', 'Верхняя ветвь'])
    ax_branch_stability.set_ylim(0, 1)

    # Теория для ветвей - убираем индивидуальные label для scatter
    if len(param_x_lower) > 0:
        ax_branch_stability.scatter(param_x_lower[ss_lower], np.ones(np.sum(ss_lower)) * y_lower_branch, marker='s', s=60, color='green')
        ax_branch_stability.scatter(param_x_lower[~ss_lower], np.ones(np.sum(~ss_lower)) * y_lower_branch, marker='s', s=60, color='red')
    if len(param_x_upper) > 0:
        ax_branch_stability.scatter(param_x_upper[ss_upper], np.ones(np.sum(ss_upper)) * y_upper_branch, marker='s', s=60, color='green')
        ax_branch_stability.scatter(param_x_upper[~ss_upper], np.ones(np.sum(~ss_upper)) * y_upper_branch, marker='s', s=60, color='red')
    if len(param_x_single) > 0: 
        ax_branch_stability.scatter(param_x_single[ss_single], np.ones(np.sum(ss_single)) * y_lower_branch, marker='s', s=60, color='green')
        ax_branch_stability.scatter(param_x_single[~ss_single], np.ones(np.sum(~ss_single)) * y_lower_branch, marker='s', s=60, color='red')
        
    # Эксперимент для ветвей - убираем индивидуальные label для scatter
    if simulation_results and any(simulation_results):
        exp_lower_stable_x, exp_lower_unstable_x = [], []
        exp_upper_stable_x, exp_upper_unstable_x = [], []

        for sim_res in simulation_results:
            param_val_sim = sim_res.get(swept_param_key) or sim_res.get('T')
            v_star_sim = sim_res.get('v_star')
            waves_sim = sim_res.get('waves_observed')

            if param_val_sim is None or v_star_sim is None or waves_sim is None:
                continue

            theoretical_states = grouped_data.get(param_val_sim, [])
            target_y_level = None
            # branch_label_suffix = '' # Больше не нужен для легенды

            if len(theoretical_states) == 1:
                target_y_level = y_lower_branch 
                # branch_label_suffix = ' (Единств.)'
            elif len(theoretical_states) >= 2:
                lower_th_v = theoretical_states[0]['v_star'] 
                upper_th_v = theoretical_states[-1]['v_star']
                if abs(v_star_sim - lower_th_v) < abs(v_star_sim - upper_th_v) or abs(v_star_sim - lower_th_v) < 1e-3: 
                    target_y_level = y_lower_branch
                    # branch_label_suffix = ' (Нижн.)'
                else:
                    target_y_level = y_upper_branch
                    # branch_label_suffix = ' (Верхн.)'
            
            if target_y_level is not None:
                if not waves_sim: 
                    if target_y_level == y_lower_branch: exp_lower_stable_x.append(param_val_sim)
                    else: exp_upper_stable_x.append(param_val_sim)
                else: 
                    if target_y_level == y_lower_branch: exp_lower_unstable_x.append(param_val_sim)
                    else: exp_upper_unstable_x.append(param_val_sim)

        if exp_lower_stable_x:
            ax_branch_stability.scatter(exp_lower_stable_x, np.ones(len(exp_lower_stable_x)) * y_lower_branch, marker='X', s=100, color='lime', edgecolor='darkgreen')
        if exp_lower_unstable_x:
            ax_branch_stability.scatter(exp_lower_unstable_x, np.ones(len(exp_lower_unstable_x)) * y_lower_branch, marker='X', s=100, color='tomato', edgecolor='darkred')
        if exp_upper_stable_x:
            ax_branch_stability.scatter(exp_upper_stable_x, np.ones(len(exp_upper_stable_x)) * y_upper_branch, marker='X', s=100, color='lime', edgecolor='darkgreen')
        if exp_upper_unstable_x:
            ax_branch_stability.scatter(exp_upper_unstable_x, np.ones(len(exp_upper_unstable_x)) * y_upper_branch, marker='X', s=100, color='tomato', edgecolor='darkred')

    # Создание кастомной легенды для ax_branch_stability
    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', label='Теория: Уст.', markersize=10, markerfacecolor='green'),
        plt.Line2D([0], [0], marker='s', color='w', label='Теория: Неуст.', markersize=10, markerfacecolor='red'),
        plt.Line2D([0], [0], marker='X', color='w', label='Эксперимент: Уст.', markersize=10, markerfacecolor='lime', markeredgecolor='darkgreen'),
        plt.Line2D([0], [0], marker='X', color='w', label='Эксперимент: Неуст.', markersize=10, markerfacecolor='tomato', markeredgecolor='darkred')
    ]
    ax_branch_stability.legend(handles=legend_handles, fontsize='small', loc='best', ncol=2)
    ax_branch_stability.grid(True, axis='x')

    # Общий заголовок и компоновка
    param_details_list = []
    for k_param, v_param in DEFAULT_IDM_PARAMS.items():
        if k_param == swept_param_key: continue
        current_val = base_params.get(k_param, v_param)
        if k_param in ['l_vehicle_length', 's0_jam_distance']:
            param_details_list.append(f"{k_param.split('_')[0]}={current_val:.1f}")
        else:
            param_details_list.append(f"{k_param.split('_')[0]}={current_val:.2f}")
    param_details = ", ".join(param_details_list)
    fig.suptitle(f'Анализ уст-ти IDM: {swept_param_label} ({fixed_condition_label})\nОст. параметры: {param_details}', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

FIXED_RING_LENGTH = 901.53 # Длина кольца для 1k.net.xml
CONFIG_NAME_FOR_SUMO = "1k" # Имя конфигурации для run_circle_simulation.py

def main(): # Обернем основной код в функцию main()
    parser = argparse.ArgumentParser(description="Анализ устойчивости EIDM и опциональный запуск SUMO симуляций.")
    
    # Изменяем путь по умолчанию на sumo-gui.exe
    default_sumo_exe_path = "sumo-gui.exe" 
    if "SUMO_HOME" in os.environ:
        # Сначала ищем sumo-gui.exe
        potential_gui_path = os.path.join(os.getenv("SUMO_HOME"), "bin", "sumo.exe")
        if os.path.isfile(potential_gui_path):
            default_sumo_exe_path = potential_gui_path
        else:
            # Если sumo-gui.exe не найден, пробуем sumo.exe как запасной вариант
            potential_cli_path = os.path.join(os.getenv("SUMO_HOME"), "bin", "sumo.exe")
            if os.path.isfile(potential_cli_path):
                default_sumo_exe_path = potential_cli_path
                print(f"ПРЕДУПРЕЖДЕНИЕ: sumo-gui.exe не найден в SUMO_HOME/bin, используется sumo.exe: {potential_cli_path}")
            else:
                print(f"ПРЕДУПРЕЖДЕНИЕ: SUMO_HOME установлен, но ни sumo-gui.exe, ни sumo.exe не найдены в {os.path.join(os.getenv('SUMO_HOME'), 'bin')}.")
    else:
        # Если SUMO_HOME не установлен, просто полагаемся на PATH для sumo-gui.exe
        pass


    parser.add_argument("--sumo-binary", type=str, default=default_sumo_exe_path, help=f"Путь к исполняемому файлу SUMO (sumo-gui.exe или sumo.exe). По умолчанию: {default_sumo_exe_path}")
    
    default_sumo_tools_path = ""
    if "SUMO_HOME" in os.environ:
        potential_tools_path = os.path.join(os.getenv("SUMO_HOME"), "tools")
        if os.path.isdir(potential_tools_path):
            default_sumo_tools_path = potential_tools_path

    parser.add_argument("--sumo-tools-dir", type=str, default=default_sumo_tools_path, help=f"Путь к директории tools в SUMO. По умолчанию: {default_sumo_tools_path}")
    parser.add_argument("--run-sumo-simulations", action="store_true", help="Запускать ли симуляции SUMO для проверки.")
    parser.add_argument("--num-vehicles-sumo", type=int, default=30, help="Количество ТС для симуляций SUMO (если не рассчитывается динамически).") # Этот аргумент теперь менее релевантен здесь
    # УДАЛЕНО: parser.add_argument("--fixed-net-file", type=str, default="config/circles/1k.net.xml", help="Путь к фиксированному .net.xml файлу для SUMO.")


    args = parser.parse_args()

    # Определяем путь к sumo-tools, если не указан явно
    if "SUMO_HOME" in os.environ and args.sumo_tools_dir == os.path.join(os.getenv("SUMO_HOME"), "tools"):
        if not os.path.isdir(args.sumo_tools_dir):
            print(f"ПРЕДУПРЕЖДЕНИЕ: Директория SUMO tools не найдена по пути из SUMO_HOME: {args.sumo_tools_dir}")
    elif not os.path.isdir(args.sumo_tools_dir) and args.sumo_tools_dir == "C:/Program Files (x86)/Eclipse/sumo-1.22.0/tools":
         pass # Не печатаем предупреждение для абсолютного пути по умолчанию, если он не существует

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
    # T_sweep_values_for_Q = np.linspace(0.8, 2.5, 50) # Старое значение
    T_sweep_values_for_Q = np.linspace(0.8, 2.5, 10) # Уменьшено для ускорения тестов с SUMO
    
    data_sweep_T_fixed_Q = collect_data_for_param_sweep(
        param_to_sweep_key='T_safe_time_headway', 
        sweep_values=T_sweep_values_for_Q,
        base_idm_params=params.copy(),
        fixed_Q=target_Q_veh_per_sec,
        verbose=False # Можно установить в True для детального лога поиска равновесий
    )
    if data_sweep_T_fixed_Q and len(data_sweep_T_fixed_Q['param_values']) > 0:
        # Здесь нужно вызвать plot_stability_for_parameter_sweep **ПЕРЕД** циклом симуляций,
        # чтобы увидеть теоретический график ДО добавления точек симуляций
        print("\\n--- Отображение теоретической устойчивости для T vs Q ---") # Добавлено для ясности
        plot_stability_for_parameter_sweep(
            data_sweep_T_fixed_Q, 
            swept_param_key='T_safe_time_headway',
            swept_param_label='Время реакции T (с)', 
            fixed_condition_label=f"Q = {target_Q_veh_per_hour:.0f} авто/час ({target_Q_veh_per_sec:.3f} авто/с)",
            base_params=params.copy(),
            simulation_results=None # Важно: передаем None, чтобы показать теоретическую устойчивость
        )

        # <<< Начало добавленного блока для запуска SUMO симуляций >>>
        if args.run_sumo_simulations:
            print("\\n--- Запуск SUMO симуляций через run_circle_simulation.py (Пример 5) ---")
            
            simulation_results_list = [] # Список для сбора результатов симуляций
            sim_params_base = params.copy() 
            vehicle_length_for_calc = sim_params_base['l_vehicle_length']
            
            num_processed_states = 0
            # Директория для всех запусков из "Примера 5"
            base_simulation_output_dir = os.path.abspath(f"results/run_circle_T_vs_Q_{CONFIG_NAME_FOR_SUMO}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(base_simulation_output_dir, exist_ok=True)

            for i in range(len(data_sweep_T_fixed_Q['param_values'])):
                current_T = data_sweep_T_fixed_Q['param_values'][i]
                current_v_e = data_sweep_T_fixed_Q['v_star'][i]
                current_s_e_net = data_sweep_T_fixed_Q['s_star_net'][i]

                if math.isnan(current_v_e) or math.isnan(current_s_e_net) or current_v_e < 0 or current_s_e_net < 0:
                    print(f"Пропуск симуляции для T={current_T:.3f}с: невалидное равновесное состояние (v_e={current_v_e:.2f}, s_e_net={current_s_e_net:.2f})")
                    continue
                
                current_total_spacing = current_s_e_net + vehicle_length_for_calc
                if current_total_spacing <= 1e-3: 
                    print(f"Предупреждение: Общий интервал (s_e_net + l_vehicle = {current_total_spacing:.3f}) слишком мал для T={current_T:.3f}. Пропуск симуляции.")
                    continue
                
                num_vehicles_for_sim = round(FIXED_RING_LENGTH / current_total_spacing)
                if num_vehicles_for_sim <= 0:
                    print(f"Предупреждение: Расчетное количество машин ({num_vehicles_for_sim}) некорректно для T={current_T:.3f}. Пропуск симуляции.")
                    continue

                num_processed_states += 1
                
                simulation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Уникальная поддиректория для этого конкретного запуска T, v_e, s_e_net
                # Эта директория будет передана в --output-dir для run_circle_simulation.py
                # Имя этой директории уже содержит всю нужную информацию + timestamp
                current_run_output_dir = os.path.join(base_simulation_output_dir, 
                                                      f"T{current_T:.3f}_v{current_v_e:.2f}_s{current_s_e_net:.2f}_N{num_vehicles_for_sim}_{simulation_timestamp}")
                os.makedirs(current_run_output_dir, exist_ok=True)

                print(f"\\nЗапуск ({num_processed_states}): T={current_T:.3f}с, v_e={current_v_e:.2f} м/с, s_e_net={current_s_e_net:.2f} м, N_veh={num_vehicles_for_sim}")
                print(f"Выходная директория симуляции: {current_run_output_dir}")

                cmd_run_sim = [
                    "python", "src/run_circle_simulation.py",
                    "--config-name", CONFIG_NAME_FOR_SUMO,
                    "--max-num-vehicles", str(num_vehicles_for_sim),
                    "--sumo-binary", args.sumo_binary,
                    "--sumo-tools-dir", args.sumo_tools_dir,
                    "--output-dir", current_run_output_dir # Передаем уникальную директорию для этого запуска
                    # Добавляем параметры IDM для симуляции
                    #"--idm-params", json.dumps({'T_safe_time_headway': current_T}) # Пока run_circle_simulation.py не поддерживает это
                ]
                
                # Временно: Обновляем IDM параметры в base_params для текущей симуляции
                # Если бы run_circle_simulation.py поддерживал --idm-params, это было бы лучше
                # Этот подход менее надежен, т.к. предполагается, что run_circle_simulation 
                # не читает IDM из .sumocfg напрямую (что он сейчас и делает).
                # Правильнее было бы передать параметры IDM в run_circle_simulation.py
                # Но для теста, оставим как есть, пока run_circle_simulation.py не модифицирован.
                
                print(f"Команда запуска симуляции: {' '.join(cmd_run_sim)}")
                sim_success = False
                try:
                    # Запускаем и ждем завершения
                    # stdout и stderr будут напечатаны run_circle_simulation.py
                    # Изменено: check=False, добавлен errors='ignore'
                    completed_process_sim = subprocess.run(cmd_run_sim, capture_output=True, text=True, errors='ignore')
                    
                    # Печатаем вывод всегда, чтобы видеть, что произошло
                    print("STDOUT (run_circle_simulation.py):")
                    print(completed_process_sim.stdout)
                    if completed_process_sim.stderr: # Печатаем stderr если он не пустой
                         print("STDERR (run_circle_simulation.py):")
                         print(completed_process_sim.stderr)
                    
                    if completed_process_sim.returncode == 0:
                        print(f"Симуляция для T={current_T:.3f} успешно завершена (run_circle_simulation.py вернул 0).")
                        sim_success = True
                    else:
                        print(f"Ошибка при выполнении симуляции для T={current_T:.3f} (run_circle_simulation.py вернул {completed_process_sim.returncode}).")
                
                except FileNotFoundError:
                    print(f"Ошибка: Не удалось найти 'python' или 'src/run_circle_simulation.py'. Убедитесь, что они в PATH или указаны правильно.")
                except subprocess.CalledProcessError as e:
                    print(f"Ошибка subprocess при запуске симуляции для T={current_T:.3f}: {e}")
                    print("STDOUT (run_circle_simulation.py - ошибка):")
                    print(e.stdout)
                    print("STDERR (run_circle_simulation.py - ошибка):")
                    print(e.stderr)
                except Exception as e:
                    print(f"Непредвиденная ошибка при запуске симуляции для T={current_T:.3f}: {e}")

                if sim_success:
                    # Имя CSV файла, как его создает run_circle_simulation.py
                    # run_circle_simulation.py создает results_dir_with_timestamp_abs, которая является current_run_output_dir.
                    # Внутри этой директории run_circle_simulation.py создает ЕЩЕ одну поддиректорию
                    # с именем f"{args.config_name}_{args.max_num_vehicles}_vehicles_{timestamp_внутри_run_circle_sim}"
                    # и уже в ней лежит fcd_output_...csv

                    # Сначала найдем эту вложенную директорию. 
                    # Мы не знаем точный timestamp из run_circle_simulation.py,
                    # но можем предположить, что там будет только одна директория, соответствующая нашему запуску,
                    # или директория, имя которой соответствует CONFIG_NAME_FOR_SUMO и num_vehicles_for_sim.
                    
                    potential_subdirs = [d for d in os.listdir(current_run_output_dir) if os.path.isdir(os.path.join(current_run_output_dir, d))]
                    actual_data_dir = None

                    if not potential_subdirs:
                        print(f"ПРЕДУПРЕЖДЕНИЕ: Не найдено поддиректорий в {current_run_output_dir}. Пропуск анализа.")
                    else:
                        # Попробуем найти директорию, которая содержит имя конфига и количество машин
                        expected_subdir_prefix = f"{CONFIG_NAME_FOR_SUMO}_{num_vehicles_for_sim}_vehicles_"
                        print(f"DEBUG: Ищем поддиректории в: {current_run_output_dir}") # Добавлено для отладки
                        print(f"DEBUG: Ожидаемый префикс поддиректории: {expected_subdir_prefix}") # Добавлено для отладки
                        found_matching_subdirs = [d for d in potential_subdirs if d.startswith(expected_subdir_prefix)]

                        if len(found_matching_subdirs) == 1:
                            actual_data_dir = os.path.join(current_run_output_dir, found_matching_subdirs[0])
                            print(f"Найдена директория с данными: {actual_data_dir}")
                        elif len(found_matching_subdirs) > 1:
                            print(f"ПРЕДУПРЕЖДЕНИЕ: Найдено несколько подходящих поддиректорий в {current_run_output_dir} с префиксом {expected_subdir_prefix}. Используется первая: {found_matching_subdirs[0]}")
                            actual_data_dir = os.path.join(current_run_output_dir, found_matching_subdirs[0])
                        elif len(potential_subdirs) == 1 : # Если только одна поддиректория, используем ее
                            actual_data_dir = os.path.join(current_run_output_dir, potential_subdirs[0])
                            print(f"Найдена одна поддиректория с данными (используем ее): {actual_data_dir}")
                        else:
                            print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось однозначно определить директорию с данными в {current_run_output_dir} (Найдено: {potential_subdirs}). Пропуск анализа.")

                    csv_file_to_analyze = None
                    if actual_data_dir:
                        print(f"DEBUG: Ищем CSV файлы в: {actual_data_dir}") # Добавлено для отладки
                        csv_files_found = [f for f in os.listdir(actual_data_dir) if f.endswith(".csv") and f.startswith("fcd_output_")]
                        if len(csv_files_found) == 1:
                            csv_file_to_analyze = os.path.join(actual_data_dir, csv_files_found[0])
                        elif len(csv_files_found) > 1:
                            print(f"ПРЕДУПРЕЖДЕНИЕ: Найдено несколько CSV файлов в {actual_data_dir}. Используется первый: {csv_files_found[0]}.")
                            csv_file_to_analyze = os.path.join(actual_data_dir, csv_files_found[0])
                        else:
                            print(f"ПРЕДУПРЕЖДЕНИЕ: CSV файл не найден в {actual_data_dir}.")
                    
                    if csv_file_to_analyze:
                        print(f"Найден CSV файл для анализа: {csv_file_to_analyze}")
                        
                        # Запускаем analyze_circle_data.py для этой директории результатов
                        # Директория результатов - это та, где лежит CSV файл
                        results_dir_for_analysis = os.path.dirname(csv_file_to_analyze)
                        analyze_cmd = [
                            "python",
                            os.path.join("src", "analyze_circle_data.py"),
                            "--results-dir", results_dir_for_analysis,
                            "--length", str(FIXED_RING_LENGTH) # Передаем длину кольца
                        ]
                        print(f"Команда запуска анализа: {' '.join(analyze_cmd)}")

                        try:
                            completed_process_analyze = subprocess.run(analyze_cmd, check=False, capture_output=True, text=True, errors='ignore') # check=False
                            print("STDOUT (analyze_circle_data.py):")
                            print(completed_process_analyze.stdout)
                            if completed_process_analyze.stderr:
                                print("STDERR (analyze_circle_data.py):")
                                print(completed_process_analyze.stderr)

                            if completed_process_analyze.returncode == 0:
                                print(f"Анализ данных для T={current_T:.3f} успешно завершен.")

                                # --- Чтение analysis_summary.json --- 
                                analysis_dir_path = None
                                # Ищем директорию analysis_* внутри actual_data_dir
                                if actual_data_dir and os.path.isdir(actual_data_dir):
                                    analysis_subdirs = [d for d in os.listdir(actual_data_dir) if d.startswith("analysis_") and os.path.isdir(os.path.join(actual_data_dir, d))]
                                    if len(analysis_subdirs) == 1:
                                        analysis_dir_path = os.path.join(actual_data_dir, analysis_subdirs[0])
                                    elif len(analysis_subdirs) > 1:
                                         print(f"Предупреждение: Найдено несколько директорий анализа в {actual_data_dir}. Используется последняя: {analysis_subdirs[-1]}")
                                         analysis_dir_path = os.path.join(actual_data_dir, analysis_subdirs[-1])
                                    else:
                                        print(f"Предупреждение: Директория анализа не найдена в {actual_data_dir}.")
                                
                                summary_file_path = None
                                if analysis_dir_path:
                                    potential_summary_file = os.path.join(analysis_dir_path, 'analysis_summary.json')
                                    if os.path.isfile(potential_summary_file):
                                        summary_file_path = potential_summary_file
                                    else:
                                        print(f"Предупреждение: Файл {potential_summary_file} не найден.")

                                if summary_file_path:
                                    try:
                                        # Добавлена проверка на существование и размер файла
                                        if os.path.getsize(summary_file_path) > 0:
                                            with open(summary_file_path, 'r', encoding='utf-8') as f_summary:
                                                analysis_summary = json.load(f_summary)
                                            waves_obs = analysis_summary.get('waves_observed')
                                            std_dev = analysis_summary.get('mean_speed_std_dev')
                                            lambda_exp_val = analysis_summary.get('lambda_exp') # Извлекаем lambda_exp

                                            if waves_obs is not None:
                                                print(f"Результат анализа симуляции: waves_observed={waves_obs} (std_dev={std_dev:.4f}, lambda_exp={lambda_exp_val})")
                                                simulation_results_list.append({
                                                    'T': current_T,
                                                    'v_star': current_v_e,
                                                    's_star_net': current_s_e_net,
                                                    'waves_observed': waves_obs,
                                                    'lambda_exp': lambda_exp_val if lambda_exp_val is not None else float('nan'), # Добавляем lambda_exp
                                                    'lambda_max_theory': data_sweep_T_fixed_Q['lambda_max_theory'][i] if 'lambda_max_theory' in data_sweep_T_fixed_Q else float('nan') # Добавляем lambda_max_theory
                                                })
                                            else:
                                                print("Предупреждение: 'waves_observed' не найдено в analysis_summary.json")
                                        
                                    except json.JSONDecodeError:
                                        print(f"Ошибка: Не удалось декодировать JSON из {summary_file_path}")
                                    except Exception as e:
                                        print(f"Ошибка при чтении/обработке {summary_file_path}: {e}")
                                # --- Конец чтения analysis_summary.json ---
                            else:
                                print(f"Ошибка при выполнении анализа данных для T={current_T:.3f} (код {completed_process_analyze.returncode}).")

                        except FileNotFoundError:
                            print(f"Ошибка: Не удалось найти 'python' или 'src/analyze_circle_data.py'.")
                        except subprocess.CalledProcessError as e:
                            print(f"Ошибка subprocess при запуске анализа для T={current_T:.3f}: {e}")
                            print("STDOUT (analyze_circle_data.py - ошибка):")
                            print(e.stdout)
                            print("STDERR (analyze_circle_data.py - ошибка):")
                            print(e.stderr)
                        except Exception as e:
                            print(f"Непредвиденная ошибка при запуске анализа для T={current_T:.3f}: {e}")
                    else: # Если csv_file_to_analyze не найден
                       print(f"Пропуск анализа для T={current_T:.3f}, так как CSV файл не найден.") 
            
            if num_processed_states == 0:
                 print("Не найдено валидных равновесных состояний для запуска SUMO симуляций в 'Примере 5' через run_circle_simulation.py.")

            # --- Перерисовываем график с результатами симуляций --- 
            if simulation_results_list:
                print("\\n--- Перерисовка графика T vs Q с результатами симуляций ---")
                plot_stability_for_parameter_sweep(
                    data_sweep_T_fixed_Q, 
                    swept_param_key='T_safe_time_headway',
                    swept_param_label='Время реакции T (с)', 
                    fixed_condition_label=f"Q = {target_Q_veh_per_hour:.0f} авто/час ({target_Q_veh_per_sec:.3f} авто/с)",
                    base_params=params.copy(),
                    simulation_results=simulation_results_list # Передаем собранные результаты
                )
            else:
                print("Не удалось собрать результаты симуляций для добавления на график.")

        # <<< Конец добавленного блока для запуска SUMO симуляций >>>
    else:
        print(f"Не удалось собрать данные для варьирования T при Q={target_Q_veh_per_hour:.0f} авто/час.")

    print("\\nАнализ завершен.") 


if __name__ == "__main__":
    main()