import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
import json
from scipy.fft import fft, fftfreq
import matplotlib.colors as mcolors

def plot_spacetime_heatmap(df, output_dir, L=None):
    """Строит x-t тепловую карту скорости."""
    if 'time' not in df.columns or 'distance' not in df.columns or 'speed' not in df.columns or 'vehicle_id' not in df.columns:
        print("Предупреждение: Недостаточно данных для построения x-t тепловой карты.")
        return

    # Если длина кольца L не задана, пытаемся оценить по максимальному расстоянию
    if L is None:
        L = df['distance'].max()
        if L == 0: L = 600 # Запасной вариант, если расстояние не посчитано

    # Для кольца используем остаток от деления на L
    df['position_on_ring'] = df['distance'] % L

    # Создаем сетку для интерполяции или используем pcolormesh
    # pcolormesh может быть проще, но требует монотонных осей
    # Попробуем сгруппировать и усреднить в бинах
    time_bins = np.linspace(df['time'].min(), df['time'].max(), 100)
    position_bins = np.linspace(0, L, 100)

    df['time_bin'] = pd.cut(df['time'], bins=time_bins, labels=False, include_lowest=True)
    df['position_bin'] = pd.cut(df['position_on_ring'], bins=position_bins, labels=False, include_lowest=True)

    heatmap_data = df.groupby(['time_bin', 'position_bin'])['speed'].mean().unstack()

    # Если есть пропуски (NaN), можно их заполнить средним или интерполировать,
    # но для начала просто отобразим как есть
    # heatmap_data = heatmap_data.fillna(df['speed'].mean())

    plt.figure(figsize=(12, 8))
    # Используем центры бинов для осей
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    position_centers = (position_bins[:-1] + position_bins[1:]) / 2
    
    # Выбираем цветовую карту (например, 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm')
    cmap = plt.get_cmap('inferno') 
    
    pcm = plt.pcolormesh(time_centers, position_centers, heatmap_data.T, shading='auto', cmap=cmap, vmin=df['speed'].min(), vmax=df['speed'].max())
    plt.colorbar(pcm, label='Скорость (м/с)')
    plt.xlabel('Время (с)')
    plt.ylabel('Положение на кольце (м)')
    plt.title('x-t Тепловая карта скорости')
    plt.savefig(os.path.join(output_dir, 'spacetime_heatmap.png'))
    plt.close()

def plot_fft_analysis(df, output_dir):
    """Выполняет FFT анализ средней скорости и строит график."""
    if 'time' not in df.columns or 'speed' not in df.columns:
        print("Предупреждение: Недостаточно данных для FFT анализа.")
        return

    # Рассчитываем среднюю скорость в каждый момент времени
    mean_speed_over_time = df.groupby('time')['speed'].mean()
    
    if len(mean_speed_over_time) < 2:
        print("Предупреждение: Недостаточно временных точек для FFT анализа.")
        return

    times = mean_speed_over_time.index.values
    speeds = mean_speed_over_time.values

    # Убираем постоянную составляющую (среднее значение)
    signal = speeds - np.mean(speeds)
    
    N = len(signal)
    # Предполагаем равномерный шаг по времени, берем средний
    T_sampling = np.mean(np.diff(times)) 
    if T_sampling <= 0 or np.isnan(T_sampling):
         print("Предупреждение: Не удалось определить шаг дискретизации для FFT.")
         return

    # Вычисляем FFT
    yf = fft(signal)
    xf = fftfreq(N, T_sampling)[:N//2] # Берем только положительные частоты

    amplitude = 2.0/N * np.abs(yf[0:N//2])

    plt.figure(figsize=(12, 6))
    plt.plot(xf, amplitude)
    plt.title('Амплитудный спектр Фурье средней скорости (исключая DC)')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.xlim(0, min(1.0, xf.max() if len(xf)>0 else 1.0)) # Ограничиваем частоту для наглядности
    plt.savefig(os.path.join(output_dir, 'fft_mean_speed.png'))
    plt.close()

def save_analysis_summary(summary_data, output_dir):
    """Сохраняет сводку анализа в JSON файл."""
    summary_file = os.path.join(output_dir, 'analysis_summary.json')
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)
        print(f"Сводка анализа сохранена в: {summary_file}")
    except Exception as e:
        print(f"Ошибка при сохранении сводки анализа в {summary_file}: {e}")

def analyze_circle_data(data_file, L=None, warmup_time=0.0):
    """Анализирует данные симуляции кругового движения из CSV файла и создает визуализации."""
    # Создаем директорию для результатов анализа, если её нет
    # Результаты анализа будут сохраняться в поддиректорию 'analysis' внутри директории с данными симуляции
    base_results_dir = os.path.dirname(data_file)
    analysis_subdir_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    analysis_dir = os.path.join(base_results_dir, analysis_subdir_name)
    os.makedirs(analysis_dir, exist_ok=True)

    # Читаем данные
    try:
        # Колонки в FCD XML, конвертированном через xml2csv.py:
        # timestep_time, vehicle_id, vehicle_x, vehicle_y, vehicle_angle, vehicle_type, vehicle_speed, vehicle_pos, vehicle_lane, vehicle_slope
        df = pd.read_csv(data_file, sep=',') 
    except pd.errors.EmptyDataError:
        print(f"Ошибка: Файл данных {data_file} пуст или некорректен.")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла {data_file}: {e}")
        return

    # Переименовываем колонки для совместимости с кодом analyze_straight_data и для понятности
    # Проверяем наличие колонок перед переименованием
    required_cols = {
        'timestep_time': 'time',
        'vehicle_id': 'vehicle_id',
        'vehicle_x': 'x',
        'vehicle_y': 'y',
        'vehicle_speed': 'speed',
        'vehicle_pos': 'lane_position',    # из FCD атрибута 'pos'
        'vehicle_odometer': 'odometer'     # из FCD атрибута 'odometer'
        # Если SUMO все еще выводит 'distance' как одометр, а не 'odometer',
        # то нужно будет использовать 'vehicle_distance': 'odometer'
    }
    cols_to_rename = {k: v for k, v in required_cols.items() if k in df.columns}
    df.rename(columns=cols_to_rename, inplace=True)

    # Проверяем, что все необходимые колонки теперь существуют (особенно 'time')
    if 'time' not in df.columns:
        print(f"Ошибка: Колонка 'time' (или 'timestep_time') отсутствует в CSV файле: {data_file}")
        print(f"Доступные колонки: {df.columns.tolist()}")
        return

    # --- Фильтрация Warmup Time --- 
    initial_rows = len(df)
    if warmup_time > 0:
        min_time = df['time'].min() # Теперь 'time' должна существовать
        df = df[df['time'] >= (min_time + warmup_time)].copy()
        print(f"Отброшено {initial_rows - len(df)} строк данных за первые {warmup_time}с (warmup).")
    if df.empty:
        print("Ошибка: После отбрасывания warmup периода не осталось данных для анализа.")
        return
    # --- Конец Фильтрации ---

    # Проверяем, что все остальные необходимые колонки существуют (после переименования и warmup)
    if not all(col in df.columns for col in ['vehicle_id', 'x', 'y', 'speed']):
        print("Ошибка: В CSV файле отсутствуют необходимые колонки (vehicle_id, x, y, speed) после переименования и warmup.")
        print(f"Доступные колонки: {df.columns.tolist()}")
        return
        
    # Убираем строки где vehicle_id это NaN, если такие есть
    df.dropna(subset=['vehicle_id'], inplace=True)

    # Сортируем данные для корректных вычислений, особенно np.diff
    df.sort_values(by=['vehicle_id', 'time'], inplace=True)

    # Вычисляем пройденное расстояние для каждого автомобиля
    vehicles = df['vehicle_id'].unique()
    df['distance'] = 0.0

    if len(vehicles) == 0:
        print(f"В файле {data_file} не найдено данных о машинах. Анализ невозможен.")
        return

    if 'odometer' in df.columns:
        print("Обнаружена колонка 'odometer'. Используется как 'distance'.")
        df['distance'] = df['odometer'].astype(float) # Убедимся, что это float для арифм. операций
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Колонка 'odometer' не найдена. \n"
              "         Расчет 'distance' будет основан на координатах x, y, как раньше.")
        # Убираем df['distance'] = 0.0 отсюда, т.к. ниже он будет создан для каждого авто
        for vehicle in vehicles:
            mask = df['vehicle_id'] == vehicle
            vehicle_data = df[mask].sort_values(by='time')
            if len(vehicle_data) > 1:
                x_coords = vehicle_data['x'].values
                y_coords = vehicle_data['y'].values
                dx = np.diff(x_coords)
                dy = np.diff(y_coords)
                distances_step = np.sqrt(dx**2 + dy**2)
                cumulative_distance = np.concatenate(([0], np.cumsum(distances_step)))
                df.loc[vehicle_data.index, 'distance'] = cumulative_distance
            elif len(vehicle_data) == 1:
                df.loc[vehicle_data.index, 'distance'] = 0.0
            else: # Если vehicle_data пуст для какого-то ID (не должно быть, если vehicles из df['vehicle_id'].unique())
                pass # или df.loc[mask, 'distance'] = 0.0, но mask будет для пустого набора

    # Применяем коррекцию 'distance' на основе начального 'lane_position'
    if 'lane_position' in df.columns and 'distance' in df.columns:
        print("Применение корректировки 'distance' на основе 'lane_position'.")
        for vehicle_id_to_adjust in vehicles: # vehicles - это df['vehicle_id'].unique()
            # Создаем маску для текущего автомобиля в оригинальном DataFrame df
            vehicle_mask_in_df = (df['vehicle_id'] == vehicle_id_to_adjust)
            
            # Получаем данные только для текущего автомобиля и сортируем их по времени
            # Это важно для корректного выбора .iloc[0]
            vehicle_specific_data_sorted = df[vehicle_mask_in_df].sort_values(by='time')

            if vehicle_specific_data_sorted.empty:
                continue

            pos_to_add = 0.0
            adjustment_applied = False

            # Убедимся, что vehicle_id_to_adjust сравнивается как строка, если это ID типа "1.0"
            current_vehicle_id_str = str(vehicle_id_to_adjust)

            if current_vehicle_id_str == '1.0':
                # Для машины '1.0', используем lane_position с первого замера (time ~ 0)
                first_record = vehicle_specific_data_sorted.iloc[0]
                pos_value = first_record['lane_position']
                pos_to_add = float(pos_value) if pd.notna(pos_value) else 0.0
                adjustment_applied = True
                print(f"Для машины {current_vehicle_id_str}: к 'distance' будет добавлен lane_position={pos_to_add:.2f} (с первого замера, time={first_record['time']:.2f}).")
            else:
                # Для других машин, используем lane_position с первого замера при time >= 1.0
                records_at_or_after_time_1 = vehicle_specific_data_sorted[vehicle_specific_data_sorted['time'] >= 1.0]
                if not records_at_or_after_time_1.empty:
                    first_record_at_or_after_1 = records_at_or_after_time_1.iloc[0]
                    pos_value = first_record_at_or_after_1['lane_position']
                    pos_to_add = float(pos_value) if pd.notna(pos_value) else 0.0
                    adjustment_applied = True
                    print(f"Для машины {current_vehicle_id_str}: к 'distance' будет добавлен lane_position={pos_to_add:.2f} (с первого замера >= 1.0с, time={first_record_at_or_after_1['time']:.2f}).")
                else:
                    print(f"Для машины {current_vehicle_id_str}: не найдено данных для времени >= 1.0с. Коррекция 'distance' на основе 'lane_position' не применена.")

            if adjustment_applied:
                # Применяем коррекцию к колонке 'distance' в оригинальном DataFrame df для всех записей этого автомобиля
                df.loc[vehicle_mask_in_df, 'distance'] = df.loc[vehicle_mask_in_df, 'distance'] + pos_to_add
    elif 'distance' not in df.columns:
         print("ПРЕДУПРЕЖДЕНИЕ: Колонка 'distance' не была создана. Коррекция невозможна.")
    else: # lane_position not in df.columns
        print("ПРЕДУПРЕЖДЕНИЕ: Колонка 'lane_position' не найдена. Коррекция 'distance' на основе 'lane_position' невозможна.")

    plots_dir = analysis_dir # Графики сохраняем в созданную директорию анализа

    # --- Расчет метрик ---
    mean_speed_std_dev = 0.0
    waves_observed = False
    if not df.empty and 'time' in df.columns and 'speed' in df.columns:
        # Группируем по времени, считаем std скорости в каждый момент, затем усредняем эти std
        std_dev_per_timestep = df.groupby('time')['speed'].std(ddof=0) # ddof=0 для std по популяции
        mean_speed_std_dev = std_dev_per_timestep.mean() 
        if np.isnan(mean_speed_std_dev): mean_speed_std_dev = 0.0 # Если всего 1 машина или 1 шаг времени
        
        # Эвристический порог для определения волн
        wave_threshold = 0.5 # м/с
        waves_observed = mean_speed_std_dev > wave_threshold
        
        print(f"Среднее стандартное отклонение скорости по времени: {mean_speed_std_dev:.4f} м/с")
        print(f"Наличие волн (std dev > {wave_threshold}): {'Да' if waves_observed else 'Нет'}")
    else:
        print("Недостаточно данных для расчета стандартного отклонения скорости.")
        
    # --- Рассчитываем lambda_exp --- 
    # Удаляем вызов lambda_exp_value = calculate_lambda_exp_std_method(df, plots_dir)
    # --- Конец расчета lambda_exp ---

    # --- Графики, аналогичные analyze_straight_data.py --- 

    # График V(t) для всех автомобилей
    plt.figure(figsize=(12, 6))
    # Цвета для выделения, если понадобится (пока не используется)
    # braking_vehicle_id = None # В круговом движении нет явного "тормозящего" по умолчанию
    # for vehicle in vehicles:
    #     vehicle_data = df[df['vehicle_id'] == vehicle]
    #     plt.plot(vehicle_data['time'], vehicle_data['speed'], 'b-', alpha=0.3 if vehicle != braking_vehicle_id else 1.0, 
    #              linewidth=1 if vehicle != braking_vehicle_id else 2, 
    #              color='red' if vehicle == braking_vehicle_id else 'blue')
    # Более простой вариант: все синие, одна машина (например, первая по ID) - красная для примера
    if len(vehicles) > 0:
        first_vehicle_id = vehicles[0]
        for i, vehicle in enumerate(vehicles):
            vehicle_data = df[df['vehicle_id'] == vehicle]
            if vehicle == first_vehicle_id:
                plt.plot(vehicle_data['time'], vehicle_data['speed'], 'r-', linewidth=1.5, label=f'Машина {vehicle} (пример)')
            elif i < 20: # Рисуем только первые 20 для наглядности, чтобы не перегружать
                 plt.plot(vehicle_data['time'], vehicle_data['speed'], 'b-', alpha=0.4, linewidth=0.8)
            elif i == 20:
                 plt.plot(vehicle_data['time'], vehicle_data['speed'], 'b-', alpha=0.4, linewidth=0.8, label='Другие машины (до 20-й)')
    plt.title('Скорость от времени V(t)')
    plt.xlabel('Время (с)')
    plt.ylabel('Скорость (м/с)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'velocity_time_all.png'))
    plt.close()

    # График S(t) для всех автомобилей
    plt.figure(figsize=(12, 6))
    if len(vehicles) > 0:
        first_vehicle_id = vehicles[0]
        for i, vehicle in enumerate(vehicles):
            vehicle_data = df[df['vehicle_id'] == vehicle]
            if vehicle == first_vehicle_id:
                plt.plot(vehicle_data['time'], vehicle_data['distance'], 'r-', linewidth=1.5, label=f'Машина {vehicle} (пример)')
            elif i < 20:
                 plt.plot(vehicle_data['time'], vehicle_data['distance'], 'b-', alpha=0.4, linewidth=0.8)
            elif i == 20:
                 plt.plot(vehicle_data['time'], vehicle_data['distance'], 'b-', alpha=0.4, linewidth=0.8, label='Другие машины (до 20-й)')
    plt.title('Пройденное расстояние от времени S(t)')
    plt.xlabel('Время (с)')
    plt.ylabel('Расстояние (м)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'distance_time_all.png'))
    plt.close()

    # График V(t) только для каждой N-ой машины (например, каждой 10-й, если их много)
    num_vehicles_total = len(vehicles)
    step = 10 if num_vehicles_total > 50 else 5 if num_vehicles_total > 20 else 1
    selected_vehicles_for_plot = vehicles[::step]

    plt.figure(figsize=(12, 6))
    colors = plt.cm.jet(np.linspace(0, 1, len(selected_vehicles_for_plot)))
    for i, vehicle in enumerate(selected_vehicles_for_plot):
        vehicle_data = df[df['vehicle_id'] == vehicle]
        plt.plot(vehicle_data['time'], vehicle_data['speed'], color=colors[i], linewidth=1.5, label=f'Машина {vehicle}')
    plt.title(f'Скорость от времени V(t) для каждой {step}-й машины')
    plt.xlabel('Время (с)')
    plt.ylabel('Скорость (м/с)')
    plt.grid(True)
    if len(selected_vehicles_for_plot) > 0 : plt.legend()
    plt.savefig(os.path.join(plots_dir, f'velocity_time_every_{step}th.png'))
    plt.close()

    # График S(t) только для каждой N-ой машины
    plt.figure(figsize=(12, 6))
    for i, vehicle in enumerate(selected_vehicles_for_plot):
        vehicle_data = df[df['vehicle_id'] == vehicle]
        plt.plot(vehicle_data['time'], vehicle_data['distance'], color=colors[i], linewidth=1.5, label=f'Машина {vehicle}')
    plt.title(f'Пройденное расстояние от времени S(t) для каждой {step}-й машины')
    plt.xlabel('Время (с)')
    plt.ylabel('Расстояние (м)')
    plt.grid(True)
    if len(selected_vehicles_for_plot) > 0 : plt.legend()
    plt.savefig(os.path.join(plots_dir, f'distance_time_every_{step}th.png'))
    plt.close()

    # График V(x) для всех автомобилей
    plt.figure(figsize=(12, 6))
    if len(vehicles) > 0:
        first_vehicle_id = vehicles[0]
        for i, vehicle in enumerate(vehicles):
            vehicle_data = df[df['vehicle_id'] == vehicle]
            if vehicle == first_vehicle_id:
                plt.plot(vehicle_data['distance'], vehicle_data['speed'], 'r-', linewidth=1.5, label=f'Машина {vehicle} (пример)')
            elif i < 20:
                 plt.plot(vehicle_data['distance'], vehicle_data['speed'], 'b-', alpha=0.4, linewidth=0.8)
            elif i == 20:
                 plt.plot(vehicle_data['distance'], vehicle_data['speed'], 'b-', alpha=0.4, linewidth=0.8, label='Другие машины (до 20-й)')
    plt.title('Скорость от положения V(x)')
    plt.xlabel('Положение на трассе (м)')
    plt.ylabel('Скорость (м/с)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'velocity_position_all.png'))
    plt.close()

    # График V(x) только для каждой N-ой машины
    plt.figure(figsize=(12, 6))
    for i, vehicle in enumerate(selected_vehicles_for_plot):
        vehicle_data = df[df['vehicle_id'] == vehicle]
        plt.plot(vehicle_data['distance'], vehicle_data['speed'], color=colors[i], linewidth=1.5, label=f'Машина {vehicle}')
    plt.title(f'Скорость от положения V(x) для каждой {step}-й машины')
    plt.xlabel('Положение на трассе (м)')
    plt.ylabel('Скорость (м/с)')
    plt.grid(True)
    if len(selected_vehicles_for_plot) > 0 : plt.legend()
    plt.savefig(os.path.join(plots_dir, f'velocity_position_every_{step}th.png'))
    plt.close()

    # График V(x) для выбранных автомобилей (N, N-1, N-2, N-4, N-9, N-19, N-49 от КОНЦА списка машин)
    # Машины в SUMO FCD обычно именуются flow_id.vehicle_running_number, например, 1.0, 1.1, ... , 1.(N-1)
    # Для простоты, будем выбирать машины по их порядковому номеру в списке `vehicles`, который отсортирован.
    # vehicles отсортированы как строки, например ['1.0', '1.1', ... '1.9', '1.10', ...]
    # Если нужно выбирать относительно последнего ID, то нужно будет правильно их отсортировать численно.
    # Предположим, что vehicle_id уже в формате 'flow.number', и сортировка строк работает корректно
    # для выбора N, N-1 и т.д. с конца.

    if num_vehicles_total > 0:
        target_indices_from_end_offsets = [1, 2, 3, 5, 10, 20, 50] # 1-й с конца, 2-й с конца и т.д.
        selected_vehicle_ids_for_special_plot = []
        
        # --- ОТЛАДКА УБРАНА --- 
        # print("Уникальные ID транспортных средств перед сортировкой:")
        # print(np.unique(vehicles))
        # --- КОНЕЦ ОТЛАДКИ ---

        # Преобразуем vehicle_id в числовой формат для корректной сортировки
        def get_sort_key(vehicle_id_str):
            s = str(vehicle_id_str)
            # Сначала пытаемся извлечь число после 'veh_'
            if "veh_" in s:
                try:
                    return float(s.split('veh_')[-1])
                except ValueError:
                    pass # Переходим к следующему правилу
            # Затем пытаемся извлечь число после последней точки
            if '.' in s:
                try:
                    return float(s.split('.')[-1])
                except ValueError:
                    pass # Переходим к следующему правилу
            # Если не подошло, пытаемся конвертировать всю строку
            try:
                return float(s)
            except ValueError:
                # Если ничего не помогло, возвращаем большое число, чтобы "неправильные" ID ушли в конец
                # или можно вернуть 0 или сам s для лексикографической сортировки таких случаев
                print(f"Предупреждение: Не удалось извлечь числовой ключ из ID '{s}'. Будет использовано как строка для сортировки этого ID.")
                return s # Лексикографическая сортировка для нераспознанных

        try:
            sorted_vehicles = sorted(vehicles, key=get_sort_key)
        except TypeError as e: # Возникает, если get_sort_key возвращает смешанные типы (числа и строки)
            print(f"Предупреждение: смешанные типы ключей при сортировке vehicle_id ({e}). Попытка полностью строковой сортировки.")
            try:
                # Попытка принудительно сделать все ключи строками, если они разные
                sorted_vehicles = sorted(vehicles, key=lambda x: str(get_sort_key(x)))
            except Exception as e_fallback:
                 print(f"Предупреждение: Полностью строковая сортировка vehicle_id также не удалась ({e_fallback}). Используется оригинальный порядок.")
                 sorted_vehicles = list(vehicles) # Худший случай - оригинальный порядок
        except ValueError as e: # От get_sort_key, если он все же выбросил ValueError (хотя не должен)
            print(f"Предупреждение: ошибка значения при сортировке vehicle_id ({e}), используется лексикографическая сортировка.")
            sorted_vehicles = sorted(vehicles)

        for offset in target_indices_from_end_offsets:
            if num_vehicles_total >= offset:
                selected_vehicle_ids_for_special_plot.append(sorted_vehicles[num_vehicles_total - offset])
        
        # Убираем дубликаты, если есть, и сохраняем порядок
        selected_vehicle_ids_for_special_plot = sorted(list(set(selected_vehicle_ids_for_special_plot)), key = lambda x: selected_vehicle_ids_for_special_plot.index(x))

        if selected_vehicle_ids_for_special_plot:
            plt.figure(figsize=(12, 6))
            plot_colors = plt.cm.viridis(np.linspace(0, 1, len(selected_vehicle_ids_for_special_plot)))
            
            for i, vehicle_id in enumerate(selected_vehicle_ids_for_special_plot):
                vehicle_data = df[df['vehicle_id'] == vehicle_id]
                if not vehicle_data.empty:
                    plt.plot(vehicle_data['distance'], vehicle_data['speed'], linestyle='-', color=plot_colors[i], linewidth=2, label=f'Машина {vehicle_id}')
            
            plt.title('Скорость от положения V(x) для выбранных машин (относительно последней)')
            # plt.xlim(min_dist, max_dist) # Можно настроить пределы по оси X, если нужно
            plt.xlabel('Положение на трассе (м)')
            plt.ylabel('Скорость (м/с)')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plots_dir, 'velocity_position_selected_cars_from_end.png'))
            plt.close()
        else:
            print("Недостаточно машин для построения графика V(x) для выбранных машин.")

    # --- Новые графики ---
    print("Построение новых графиков...")
    try:
        plot_spacetime_heatmap(df.copy(), plots_dir, L=L) # Передаем L
    except Exception as e:
        print(f"Ошибка при построении x-t тепловой карты: {e}")
    
    try:
        plot_fft_analysis(df.copy(), plots_dir)
    except Exception as e:
        print(f"Ошибка при построении FFT анализа: {e}")

    # --- Сохранение сводки ---
    summary_data = {
        'data_file': os.path.basename(data_file),
        'ring_length_used': L if L is not None else (df['distance'].max() if not df.empty and 'distance' in df.columns else 0),
        'mean_speed_std_dev': mean_speed_std_dev,
        'waves_observed_threshold': wave_threshold,
        # Преобразуем numpy.bool_ в стандартный bool для JSON
        'waves_observed': bool(waves_observed),
        'analysis_timestamp': datetime.now().isoformat()
        # Дополнительные параметры (T, v_e, s_e_net, L) нужно будет добавить извне, 
        # когда этот скрипт вызывается из другого скрипта, который их знает.
    }
    save_analysis_summary(summary_data, analysis_dir)

    print(f"Анализ завершен. Результаты сохранены в директории: {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description='Анализ FCD данных для кольцевой трассы SUMO и создание графиков.')
    parser.add_argument('--file', type=str, required=True, help='Путь к CSV файлу с FCD данными.')
    parser.add_argument('--length', '-L', type=float, default=None, help='Длина кольцевой трассы (м). Если не задано, используется max(distance).')
    parser.add_argument('--warmup-time', type=float, default=150.0, help='Время прогрева симуляции (с), которое нужно отбросить в начале.')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Ошибка: Файл {args.file} не найден.")
        return
    
    analyze_circle_data(args.file, L=args.length, warmup_time=args.warmup_time)

if __name__ == "__main__":
    main() 