import numpy as np
import matplotlib.pyplot as plt
import os
from run_straight import run_simulation
from generate_straight_rou import generate_straight_rou
import pandas as pd
from analyze_straight_data import analyze_straight_data

def analyze_experiment(results_dir, tau_reaction, q_fixed, s0, a, b, v0, delta, eidm_params):
    # Анализируем экспериментальные данные и считаем Re(lambda)
    # tau_reaction - это время реакции водителя (эквивалент step_length из старой версии, если он означал время реакции)
    density_flow_file = os.path.join(results_dir, 'density_flow_data.csv')
    if not os.path.exists(density_flow_file):
        print(f"Файл {density_flow_file} не найден!")
        return None, None
    df = pd.read_csv(density_flow_file)
    q_exp = df['flow'].mean()
    v_exp = df['mean_speed'].mean()
    rho_exp = df['density'].mean()

    if pd.isna(rho_exp) or pd.isna(q_exp) or pd.isna(v_exp): # Исправлено на pd.isna
        print("Ошибка анализа данных: NaN в значениях потока, скорости или плотности.")
        return None, None
    
    # Проверка на случай, если rho_exp или v_exp равны нулю, чтобы избежать деления на ноль
    if rho_exp == 0 or v_exp == 0:
        print("Ошибка анализа данных: плотность или скорость равны нулю, невозможно вычислить T_exp.")
        return None, None

    # Восстанавливаем T_exp по формуле: T_headway_exp = (1/rho_exp - s0)/v_exp 
    # Здесь 1/rho_exp это S_exp (полный интервал). s_exp_gap = S_exp - L
    # T_headway_exp = (S_exp - L - s0) / v_exp
    S_exp = 1/rho_exp
    s_exp_gap = S_exp - eidm_params['length_mean'] 
    
    if v_exp == 0: # Дополнительная проверка, хотя уже покрыта выше
        T_headway_exp = np.nan # или другое значение по умолчанию
    else:
        T_headway_exp = (s_exp_gap - s0) / v_exp

    # Расчет Re_lambda на основе экспериментальных v_e_exp и s_e_exp_gap
    # s_e_exp_gap уже вычислен как s_exp_gap
    v_e_exp = v_exp 

    # Коэффициенты для формулы устойчивости (используя tau_reaction)
    # Убедимся, что s_exp_gap не слишком мал (например, > 0)
    if s_exp_gap <= 1e-6: # Малый или отрицательный чистый промежуток
        print(f"Предупреждение: Экспериментальный чистый промежуток s_exp_gap ({s_exp_gap:.2f}) слишком мал. Re_lambda не будет рассчитан.")
        return T_headway_exp, np.nan

    term1 = (v_e_exp/v0)**delta # delta - параметр EIDM
    # Обратите внимание: T_headway_exp здесь это T из формул s_0 + vT
    A_stab = -a * (delta/v_e_exp * term1 + 2*T_headway_exp*(s0 + T_headway_exp*v_e_exp)/s_exp_gap**2)
    B_stab = a * v_e_exp * (s0 + T_headway_exp*v_e_exp) / (np.sqrt(a*b) * s_exp_gap**2)
    C_stab = 2 * a * (s0 + T_headway_exp*v_e_exp)**2 / s_exp_gap**3
    
    D_stab = A_stab + 2*B_stab
    # Проверка на случай, если D_stab^4 + 16*C_stab^2 < 0 (хотя маловероятно с квадратами)
    sqrt_term_val = D_stab**4 + 16*C_stab**2
    if sqrt_term_val < 0:
        print(f"Предупреждение: Подкоренное выражение для omega ({sqrt_term_val:.2f}) отрицательно. Re_lambda не будет рассчитан.")
        return T_headway_exp, np.nan
        
    omega = np.sqrt(0.5*(D_stab**2 + np.sqrt(sqrt_term_val)))
    phi = omega * tau_reaction # tau_reaction - время реакции водителя
    Re_lambda = D_stab * np.cos(phi) - omega * np.sin(phi)
    
    return T_headway_exp, Re_lambda

def main():
    # Фиксированные параметры модели и симуляции
    driver_reaction_time = 0.5  # Время реакции водителя (tau в аналитике, stepping в EIDM)
    simulation_step_duration = 0.1 # Шаг времени симуляции SUMO
    
    N = 110   # Количество автомобилей
    s0 = 2.0  # Минимальный промежуток (м)
    v0 = 33   # Желаемая скорость на свободной дороге (м/с)
    a = 0.3   # Ускорение (м/с^2)
    b = 4.5   # Комфортное замедление (м/с^2)
    delta = 4.0 # Экспонента ускорения в IDM
    L = 5.0   # Длина автомобиля (м)

    print("--- Ручной запуск симуляции EIDM в SUMO ---")
    # Ручной ввод v_e и s_e_gap
    try:
        v_e_manual = float(input(f"Введите равновесную скорость v_e (м/с, например, {0.8*v0:.1f}): "))
        # s_e_gap это чистый промежуток s*_0 = s_0 + v_e * T_headway
        s_e_gap_manual = float(input(f"Введите равновесный чистый промежуток s_e_gap (м, должен быть >= s0={s0}, например, {s0 + 1.0*v_e_manual:.1f} для T_headway=1.0с): "))
    except ValueError:
        print("Ошибка: введите числовые значения.")
        return

    if v_e_manual <= 0:
        print("Ошибка: v_e должна быть положительной.")
        return
    if v_e_manual > v0:
        print(f"Предупреждение: Заданная скорость v_e ({v_e_manual:.1f} м/с) больше максимальной скорости модели v0 ({v0:.1f} м/с).")
        # Продолжаем, но это может быть нефизично для равновесия EIDM
        
    if s_e_gap_manual < s0:
        print(f"Ошибка: s_e_gap ({s_e_gap_manual}м) должен быть не меньше s0 ({s0}м).")
        return

    # Вычисляем T_headway (желаемый временной интервал)
    # и q (плотность потока)
    # s_e_gap = s0 + v_e * T_headway  => T_headway = (s_e_gap - s0) / v_e
    if v_e_manual == 0: # Уже проверено, что >0, но для полноты
        print("Ошибка: v_e не может быть 0 для расчета T_headway.")
        return
    T_headway_manual = 1
    
    # S_e_total - полный интервал от бампера до бампера (headway distance)
    S_e_total_manual = s_e_gap_manual + L 
    if S_e_total_manual <= 0: # Маловероятно, если L > 0
        print("Ошибка: Общий интервал S_e_total_manual должен быть положительным.")
        return
        
    q_manual = v_e_manual / S_e_total_manual

    print(f"\n=== Параметры для ручного запуска симуляции ===")
    print(f"Заданная равновесная скорость (v_e): {v_e_manual:.4f} м/с")
    print(f"Заданный чистый промежуток (s_e_gap): {s_e_gap_manual:.4f} м")
    print(f"Длина автомобиля (L): {L:.4f} м")
    print(f"Мин. промежуток (s0): {s0:.4f} м")
    print(f"Время реакции водителя (driver_reaction_time): {driver_reaction_time:.4f} сек")
    print(f"Шаг симуляции SUMO (simulation_step_duration): {simulation_step_duration:.4f} сек")
    print(f"--- Вычисленные параметры для EIDM ---")
    print(f"Желаемый временной интервал (T_headway, tau_mean в EIDM): {T_headway_manual:.4f} сек")
    print(f"Полный интервал (S_e_total = s_e_gap + L): {S_e_total_manual:.4f} м")
    print(f"Плотность потока (q = v_e / S_e_total): {q_manual:.4f} авт/сек (или {q_manual*3600:.0f} авт/час)")
    
    eidm_params = {
        'accel_mean': a,
        'accel_std': 0,
        'decel_mean': b,
        'decel_std': 0,
        'sigma_mean': 0, 
        'sigma_std': 0,
        'tau_mean': T_headway_manual,  # Желаемый временной интервал T в EIDM
        'tau_std': 0,
        'delta_mean': delta,          # Экспонента IDM
        'delta_std': 0,
        'stepping_mean': driver_reaction_time, # Время реакции водителя tau_reaction
        'stepping_std': 0,
        'length_mean': L,             # Длина автомобиля
        'length_std': 0,
        'min_gap_mean': s0,           # Минимальный промежуток s0
        'min_gap_std': 0,
        'max_speed_mean': v0,         # Максимальная скорость v0 (желаемая на свободной дороге)
        'max_speed_std': 0.0
    }

    # Генерируем rou.xml
    # equilibrium_spacing в generate_straight_rou это чистый промежуток s_e_gap
    print("\nГенерация файла маршрутов (rou.xml)...")
    generate_straight_rou(N, eidm_params, q_manual, start_speed=v_e_manual, equilibrium_spacing=s_e_gap_manual)

    # Запускаем симуляцию SUMO
    print("\nЗапуск симуляции SUMO...")
    results_dir = run_simulation(N, eidm_params, q_manual, 
                                 step_length=simulation_step_duration,
                                 v_e=v_e_manual)
    if results_dir is None:
        print("Ошибка при запуске симуляции SUMO. Проверьте лог выше.")
        return
        
    print(f"Результаты симуляции сохранены в: {results_dir}")

    # Анализируем экспериментальные данные для вычисления Re(lambda)
    print("\nАнализ результатов эксперимента (расчет T_exp, Re_lambda_exp)...")
    T_exp_val, Re_lambda_exp_val = analyze_experiment(results_dir, driver_reaction_time, q_manual, s0, a, b, v0, delta, eidm_params)
    if T_exp_val is not None and Re_lambda_exp_val is not None:
        print(f"  Экспериментальный временной интервал T_exp (из данных SUMO): {T_exp_val:.4f} сек")
        print(f"  Экспериментальный показатель устойчивости Re_lambda_exp: {Re_lambda_exp_val:.4f}")
        if not np.isnan(Re_lambda_exp_val) and Re_lambda_exp_val > 0:
            print("  ВНИМАНИЕ: Re_lambda_exp > 0, поток НЕУСТОЙЧИВ по линейной теории.")
        elif not np.isnan(Re_lambda_exp_val):
            print("  Re_lambda_exp <= 0, поток УСТОЙЧИВ по линейной теории.")
    else:
        print("  Не удалось рассчитать T_exp и/или Re_lambda_exp из данных SUMO.")

    # Запуск анализа траекторий и построения графиков из analyze_straight_data
    sim_data_file = os.path.join(results_dir, 'simulation_data.csv')
    if os.path.exists(sim_data_file):
        print(f"\nЗапуск детального анализа траекторий из {sim_data_file}...")
        analyze_straight_data(sim_data_file)
        print(f"Графики анализа траекторий должны быть сохранены в {os.path.join(results_dir, 'plots')}")
    else:
        print(f"\nФайл данных симуляции {sim_data_file} не найден. Пропуск детального анализа траекторий.")
        
    print("\nСкрипт завершил работу.")

if __name__ == "__main__":
    main() 