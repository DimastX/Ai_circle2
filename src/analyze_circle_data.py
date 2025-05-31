import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
import json
from scipy.fft import fft, fftfreq
import matplotlib.colors as mcolors

# Импорт функции для анализа волн
import eidm_stability_analysis # Предполагаем, что он в src

def plot_spacetime_heatmap(df, output_dir, L=None, ax=None):
    """Строит x-t тепловую карту скорости. Может использовать существующий ax."""
    if 'time' not in df.columns or 'distance' not in df.columns or 'speed' not in df.columns or 'vehicle_id' not in df.columns:
        print("Предупреждение: Недостаточно данных для построения x-t тепловой карты.")
        return None # Возвращаем None, если не удалось построить

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

    if ax is None:
        fig, current_ax = plt.subplots(figsize=(12, 8))
    else:
        current_ax = ax
        fig = current_ax.figure

    # Используем центры бинов для осей
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    position_centers = (position_bins[:-1] + position_bins[1:]) / 2
    
    # Выбираем цветовую карту (например, 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm')
    cmap = plt.get_cmap('inferno') 
    
    pcm = current_ax.pcolormesh(time_centers, position_centers, heatmap_data.T, shading='auto', cmap=cmap, vmin=df['speed'].min(), vmax=df['speed'].max())
    
    # Добавляем colorbar, только если мы создали фигуру здесь
    if ax is None:
        fig.colorbar(pcm, ax=current_ax, label='Скорость (м/с)')
    else: # Если ax передан, предполагаем, что colorbar будет управляться извне или не нужен на каждом подграфике
        pass 

    current_ax.set_xlabel('Время (с)')
    current_ax.set_ylabel('Положение на кольце (м)')
    current_ax.set_title('x-t Тепловая карта скорости (V(x,t) из FCD)')
    
    if ax is None: # Сохраняем, только если это основной вызов для этой функции
        plt.savefig(os.path.join(output_dir, 'spacetime_heatmap_fcd.png'))
        plt.close(fig)
    return current_ax # Возвращаем axes для дальнейшего использования

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

def analyze_circle_data(results_dir, L=None, warmup_time=0.0):
    """Анализирует данные симуляции кругового движения из CSV файла и данных детекторов, создает визуализации."""
    # Создаем директорию для результатов анализа, если её нет
    # Результаты анализа будут сохраняться в поддиректорию 'analysis' внутри директории с данными симуляции
    # base_results_dir = os.path.dirname(data_file) # Старый подход
    analysis_subdir_name = f"analysis_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    analysis_plot_dir = os.path.join(results_dir, analysis_subdir_name) # Сохраняем графики в подпапку results_dir
    os.makedirs(analysis_plot_dir, exist_ok=True)

    # --- Загрузка данных детекторов --- 
    V_data_detectors, N_data_detectors, Q_data_detectors, rho_data_detectors, cameras_coords, Ts_detector = None, None, None, None, None, None
    detector_data_loaded = False
    rt_wave_event_times = [] # <--- Список для времен событий из rt_detected_wave_events.csv

    try:
        v_data_path = os.path.join(results_dir, "V_data_detectors.npy")
        n_data_path = os.path.join(results_dir, "N_data_detectors.npy")
        q_data_path = os.path.join(results_dir, "Q_data_detectors.npy") # <--- Путь к Q_data
        rho_data_path = os.path.join(results_dir, "rho_data_detectors.npy") # <--- Путь к rho_data
        cameras_coords_path = os.path.join(results_dir, "cameras_coords.npy")
        ts_detector_path = os.path.join(results_dir, "Ts_detector.txt")

        if os.path.exists(v_data_path) and os.path.exists(n_data_path) and \
           os.path.exists(q_data_path) and os.path.exists(rho_data_path) and \
           os.path.exists(cameras_coords_path) and os.path.exists(ts_detector_path):
            
            V_data_detectors = np.load(v_data_path)
            N_data_detectors = np.load(n_data_path)
            Q_data_detectors = np.load(q_data_path)   # <--- Загрузка Q_data
            rho_data_detectors = np.load(rho_data_path) # <--- Загрузка rho_data
            cameras_coords = np.load(cameras_coords_path)
            with open(ts_detector_path, 'r') as f_ts:
                Ts_detector = float(f_ts.read().strip())
            
            print("Данные детекторов (V, N, Q, rho, coords, Ts) успешно загружены.")
            print(f"  V_data shape: {V_data_detectors.shape}, N_data shape: {N_data_detectors.shape}, Q_data shape: {Q_data_detectors.shape}, rho_data shape: {rho_data_detectors.shape}")
            print(f"  cameras_coords: {cameras_coords}, Ts: {Ts_detector}")
            detector_data_loaded = True
        else:
            print("Предупреждение: Один или несколько файлов данных детекторов (.npy, .txt) не найдены в директории:")
            print(f"  V_path: {v_data_path} (exists: {os.path.exists(v_data_path)})")
            print(f"  N_path: {n_data_path} (exists: {os.path.exists(n_data_path)})")
            print(f"  Q_path: {q_data_path} (exists: {os.path.exists(q_data_path)})")
            print(f"  rho_path: {rho_data_path} (exists: {os.path.exists(rho_data_path)})")
            print(f"  coords_path: {cameras_coords_path} (exists: {os.path.exists(cameras_coords_path)})")
            print(f"  Ts_path: {ts_detector_path} (exists: {os.path.exists(ts_detector_path)})")
            print("Анализ stop-and-go волн и графики Q/rho по данным детекторов могут быть пропущены или неполны.")

        # +++ Загрузка времен событий из rt_detected_wave_events.csv +++
        rt_events_csv_path = os.path.join(results_dir, "rt_detected_wave_events.csv")
        if os.path.exists(rt_events_csv_path):
            try:
                rt_events_df = pd.read_csv(rt_events_csv_path)
                if 't_event_s' in rt_events_df.columns:
                    rt_wave_event_times = rt_events_df['t_event_s'].tolist()
                    print(f"Загружено {len(rt_wave_event_times)} времен событий из {rt_events_csv_path}")
                else:
                    print(f"Предупреждение: столбец 't_event_s' не найден в {rt_events_csv_path}")
            except Exception as e_csv_load:
                print(f"Ошибка при загрузке {rt_events_csv_path}: {e_csv_load}")
        else:
            print(f"Файл {rt_events_csv_path} не найден. Аннотации событий RT не будут добавлены.")
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    except Exception as e:
        print(f"Ошибка при загрузке данных детекторов: {e}")
        # Здесь могла быть дополнительная обработка или логирование специфических ошибок
        # для V, N, Q, rho, coords, Ts по отдельности, если бы они возникали.
        # Поскольку сейчас это общий блок, и основная информация об ошибке уже выведена,
        # добавляем pass, чтобы удовлетворить синтаксис, если других действий не требуется.
        pass
    # --- Конец загрузки данных детекторов ---

    # --- Поиск и чтение FCD CSV файла --- 
    fcd_csv_file_path = None
    try:
        # Ищем файл, который соответствует шаблону fcd_output_*.csv
        for fname in os.listdir(results_dir):
            if fname.startswith("fcd_output_") and fname.endswith(".csv"):
                fcd_csv_file_path = os.path.join(results_dir, fname)
                print(f"Найден FCD CSV файл: {fcd_csv_file_path}")
                break
        if not fcd_csv_file_path:
            print(f"Ошибка: FCD CSV файл (fcd_output_*.csv) не найден в директории {results_dir}.")
            # Если нет FCD, то основной анализ невозможен, но анализ волн по детекторам еще может быть
            # return # Решаем, выходить ли, или пытаться работать только с детекторами
    except Exception as e:
        print(f"Ошибка при поиске FCD CSV файла в {results_dir}: {e}")
        # return

    df = None
    if fcd_csv_file_path and os.path.exists(fcd_csv_file_path):
        try:
            df = pd.read_csv(fcd_csv_file_path, sep=',') 
            print(f"FCD CSV файл {fcd_csv_file_path} успешно загружен.")
        except pd.errors.EmptyDataError:
            print(f"Ошибка: Файл данных FCD CSV {fcd_csv_file_path} пуст или некорректен.")
            df = None # Убедимся, что df это None
        except Exception as e:
            print(f"Ошибка при чтении файла FCD CSV {fcd_csv_file_path}: {e}")
            df = None
    else:
        print("FCD CSV файл не загружен. Анализ на основе FCD будет пропущен.")

    # Если нет ни FCD данных, ни данных детекторов, то анализировать нечего
    if df is None and not detector_data_loaded:
        print("Отсутствуют как FCD данные, так и данные детекторов. Анализ невозможен.")
        return
    # --- Конец чтения FCD CSV ---

    # Дальнейший код будет использовать analysis_plot_dir для сохранения графиков
    # и df для FCD данных, если они есть, и V_data_detectors и т.д. для данных детекторов

    # Переименовываем колонки для совместимости, если df загружен
    if df is not None:
        # Колонки в FCD XML, конвертированном через xml2csv.py:
        # timestep_time, vehicle_id, vehicle_x, vehicle_y, vehicle_angle, vehicle_type, vehicle_speed, vehicle_pos, vehicle_lane, vehicle_slope
        df.rename(columns={
            'timestep_time': 'time',
            'vehicle_id': 'vehicle_id',
            'vehicle_x': 'x',
            'vehicle_y': 'y',
            'vehicle_speed': 'speed',
            'vehicle_pos': 'lane_position',    # из FCD атрибута 'pos'
            'vehicle_odometer': 'odometer'     # из FCD атрибута 'odometer'
            # Если SUMO все еще выводит 'distance' как одометр, а не 'odometer',
            # то нужно будет использовать 'vehicle_distance': 'odometer'
        }, inplace=True)

        # Проверяем, что все необходимые колонки теперь существуют (особенно 'time')
        if 'time' not in df.columns:
            print(f"Ошибка: Колонка 'time' (или 'timestep_time') отсутствует в CSV файле: {fcd_csv_file_path}")
            print(f"Доступные колонки: {df.columns.tolist()}")
            return

    # --- Фильтрация Warmup Time --- 
    if df is not None and not df.empty and 'time' in df.columns: # Убедимся, что df пригоден для использования
        initial_rows_fcd = len(df)
        if warmup_time < 0:
            # Фильтруем строки FCD, где время меньше warmup_time.
            # Это предполагает, что warmup_time - это абсолютное значение от начала симуляции (0с).
            df = df[df['time'] >= warmup_time].copy()
            print(f"Отброшено {initial_rows_fcd - len(df)} строк FCD данных за первые {warmup_time}с (warmup).")
            if df.empty:
                print("ПРЕДУПРЕЖДЕНИЕ: После отбрасывания warmup периода не осталось FCD данных для анализа.")
                # Не выходим, так как анализ по данным детекторов (если они есть) еще может быть выполнен.
    elif df is None or df.empty:
        print("FCD данные отсутствуют или пусты. Фильтрация warmup для FCD не применяется.")
    # --- Конец Фильтрации ---

    # Проверяем, что все остальные необходимые колонки существуют (после переименования и warmup)
    if df is not None and not df.empty: # Продолжаем проверки только если df все еще содержит данные
        if not all(col in df.columns for col in ['vehicle_id', 'x', 'y', 'speed']):
            print("Ошибка: В CSV файле отсутствуют необходимые колонки (vehicle_id, x, y, speed) после переименования и warmup.")
            print(f"Доступные колонки: {df.columns.tolist()}")
            df = None # Считаем df непригодным для дальнейшего FCD анализа
        else:
            # Убираем строки где vehicle_id это NaN, если такие есть
            df.dropna(subset=['vehicle_id'], inplace=True)
            # Сортируем данные для корректных вычислений, особенно np.diff
            df.sort_values(by=['vehicle_id', 'time'], inplace=True)
            if df.empty: # Могло стать пустым после dropna
                 print("ПРЕДУПРЕЖДЕНИЕ: FCD данные стали пустыми после dropna(subset=['vehicle_id']).")
                 df = None
    # Если df стал None, последующие блоки FCD анализа будут пропускаться

    # Вычисляем пройденное расстояние для каждого автомобиля
    vehicles = df['vehicle_id'].unique()
    df['distance'] = 0.0

    if len(vehicles) == 0:
        print(f"В файле {fcd_csv_file_path} не найдено данных о машинах. Анализ невозможен.")
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

    plots_dir = analysis_plot_dir # Графики сохраняем в созданную директорию анализа

    # --- Расчет метрик ---
    mean_speed_std_dev = 0.0
    waves_observed = False
    mean_speed_overall = 0.0  # Добавляем среднюю скорость
    if not df.empty and 'time' in df.columns and 'speed' in df.columns:
        # Группируем по времени, считаем std скорости в каждый момент, затем усредняем эти std
        std_dev_per_timestep = df.groupby('time')['speed'].std(ddof=0) # ddof=0 для std по популяции
        mean_speed_std_dev = std_dev_per_timestep.mean() 
        if np.isnan(mean_speed_std_dev): mean_speed_std_dev = 0.0 # Если всего 1 машина или 1 шаг времени
        
        # Рассчитываем среднюю скорость за всю симуляцию
        mean_speed_overall = df['speed'].mean()
        if np.isnan(mean_speed_overall): mean_speed_overall = 0.0
        
        # Эвристический порог для определения волн
        wave_threshold = 0.5 # м/с
        waves_observed = mean_speed_std_dev > wave_threshold
        
        print(f"Среднее стандартное отклонение скорости по времени: {mean_speed_std_dev:.4f} м/с")
        print(f"Средняя скорость за всю симуляцию: {mean_speed_overall:.4f} м/с")
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
    
    # Добавляем VSL данные, если доступны
    vsl_log_path = os.path.join(results_dir, "vsl_controller_log.csv")
    if os.path.exists(vsl_log_path):
        try:
            vsl_df = pd.read_csv(vsl_log_path)
            if 'sim_time_s' in vsl_df.columns and 'vsl_applied_speed_m_s' in vsl_df.columns:
                vsl_time = vsl_df['sim_time_s']
                vsl_speed = vsl_df['vsl_applied_speed_m_s']
                plt.plot(vsl_time, vsl_speed, 'g-', linewidth=2, label='VSL скорость', alpha=0.8)
                print(f"Добавлена VSL скорость на график V(t) ({len(vsl_df)} точек)")
        except Exception as e:
            print(f"Ошибка при загрузке VSL данных: {e}")
    
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

    # --- Вызов анализа stop-and-go волн, если данные детекторов загружены ---
    # ЗАКОММЕНТИРОВАНО: старый анализ волн и аннотации
    # if detector_data_loaded:
    #     print("\n--- Анализ Stop-and-Go волн по данным детекторов ---")
    #     
    #     fig_vx_det, ax_vx_det = plt.subplots(figsize=(10, 6))
    #     fig_vt_det, ax_vt_det = plt.subplots(figsize=(10, 6))
    #     fig_heatmap_det, ax_heatmap_det = plt.subplots(figsize=(12, 8))
    #
    #     # 1. График V(x) по данным детекторов (средняя скорость на каждой камере)
    #     # if V_data_detectors is not None and cameras_coords is not None:
    #     #     mean_speed_per_camera = np.nanmean(V_data_detectors, axis=1)
    #     #     ax_vx_det.plot(cameras_coords, mean_speed_per_camera, marker='o', linestyle='-', label="V_avg(x) по детекторам")
    #     #     ax_vx_det.set_xlabel("Координата X (м)")
    #     #     ax_vx_det.set_ylabel("Средняя скорость (м/с)")
    #     #     ax_vx_det.set_title("Средняя скорость по координате (детекторы)")
    #     #     ax_vx_det.grid(True)
    #     #     ax_vx_det.legend()
    #
    #     # 2. График V(t) для первой камеры (детекторы)
    #     # if V_data_detectors is not None and Ts_detector is not None and V_data_detectors.shape[0] > 0:
    #     #     time_axis_det = np.arange(V_data_detectors.shape[1]) * Ts_detector
    #     #     ax_vt_det.plot(time_axis_det, V_data_detectors[0, :], linestyle='-', label=f"V(t) на камере 0 (детектор)")
    #     #     ax_vt_det.set_xlabel("Время (с)")
    #     #     ax_vt_det.set_ylabel("Средняя скорость (м/с)")
    #     #     ax_vt_det.set_title("Средняя скорость на первой камере (детектор)")
    #     #     ax_vt_det.grid(True)
    #     #     ax_vt_det.legend()
    # 
    #     # 3. Тепловая карта V(x,t) по данным детекторов
    #     # if V_data_detectors is not None and cameras_coords is not None and Ts_detector is not None:
    #     #     time_axis_heatmap_det = np.arange(V_data_detectors.shape[1]) * Ts_detector
    #     #     X_heatmap_det, T_heatmap_det = np.meshgrid(cameras_coords, time_axis_heatmap_det)
    #     #     # Для pcolormesh V_data должен быть (num_timesteps, num_cameras)
    #     #     im_heatmap = ax_heatmap_det.pcolormesh(T_heatmap_det, X_heatmap_det, V_data_detectors.T, shading='auto', cmap='viridis')
    #     #     ax_heatmap_det.set_xlabel("Время (с)")
    #     #     ax_heatmap_det.set_ylabel("Координата X (м)")
    #     #     ax_heatmap_det.set_title("Пространственно-временная диаграмма V(x,t) (детекторы)")
    #     #     plt.colorbar(im_heatmap, ax=ax_heatmap_det, label='Скорость (м/с)')
    #     
    #     # Вызов функции обнаружения волн
    #     # try:
    #     #     events_list = es.detect_stop_and_go_waves(
    #     #         cameras_coords=cameras_coords,
    #     #         Ts=Ts_detector,
    #     #         V_data=V_data_detectors,
    #     #         N_data=N_data_detectors,
    #     #         Q_data=Q_data_detectors, # Передаем Q
    #     #         rho_data=rho_data_detectors, # Передаем rho
    #     #         axes_vx=ax_vx_det,
    #     #         axes_vt=ax_vt_det,
    #     #         axes_heatmap=ax_heatmap_det,
    #     #         output_csv_path=os.path.join(analysis_plot_dir, "detected_wave_events_from_postprocessing.csv")
    #     #     )
    #     #     print(f"Функция detect_stop_and_go_waves (постобработка) завершена. Найдено событий: {len(events_list)}")
    #     # except Exception as e_wave:
    #     #     print(f"Ошибка при вызове detect_stop_and_go_waves: {e_wave}")
    #
    #     # Сохранение графиков с аннотациями от detect_stop_and_go_waves
    #     # fig_vx_det.savefig(os.path.join(analysis_plot_dir, "vx_plot_detectors_annotated.png"))
    #     # plt.close(fig_vx_det)
    #     # fig_vt_det.savefig(os.path.join(analysis_plot_dir, "vt_plot_detectors_annotated.png"))
    #     # plt.close(fig_vt_det)
    #     # fig_heatmap_det.savefig(os.path.join(analysis_plot_dir, "heatmap_detectors_annotated.png"))
    #     # plt.close(fig_heatmap_det)
    # else:
    #     print("Данные детекторов не загружены, анализ stop-and-go волн по ним не будет выполнен.")

    # --- Построение графиков Q(x), Q(t), rho(x), rho(t) --- 
    if detector_data_loaded:
        print("\n--- Построение графика rho(t) по данным детекторов ---")

        # +++ Восстановление и модификация кода для тепловой карты V(x,t) по данным детекторов +++
        # ЗАКОММЕНТИРОВАНО ПО ЗАПРОСУ ПОЛЬЗОВАТЕЛЯ
        # fig_heatmap_v_det, ax_heatmap_v_det = plt.subplots(figsize=(12, 8))
        # if V_data_detectors is not None and V_data_detectors.size > 0 and cameras_coords is not None and Ts_detector is not None:
        #     time_steps_heatmap_v = V_data_detectors.shape[1]
        #     time_axis_heatmap_v = np.arange(time_steps_heatmap_v + 1) * Ts_detector 
        #     sorted_cam_indices = np.argsort(cameras_coords)
        #     sorted_cameras_coords = cameras_coords[sorted_cam_indices]
        #     sorted_V_data_detectors = V_data_detectors[sorted_cam_indices, :]
        #     pcm_heatmap_v = ax_heatmap_v_det.pcolormesh(
        #         time_axis_heatmap_v,
        #         sorted_cameras_coords, 
        #         sorted_V_data_detectors, 
        #         shading='auto', 
        #         cmap='inferno', 
        #         vmin=np.nanmin(V_data_detectors), 
        #         vmax=np.nanmax(V_data_detectors)
        #     )
        #     fig_heatmap_v_det.colorbar(pcm_heatmap_v, ax=ax_heatmap_v_det, label='Скорость V (м/с)')
        #     ax_heatmap_v_det.set_xlabel(f"Время t (с) (Ts={Ts_detector:.2f}c)")
        #     ax_heatmap_v_det.set_ylabel("Положение камеры на кольце, x (м)")
        #     ax_heatmap_v_det.set_title("x-t Тепловая карта скорости V(x,t) (данные детекторов)")
        #     if rt_wave_event_times:
        #         added_label_v_heatmap = False
        #         for t_event in rt_wave_event_times:
        #             label = "RT Wave Event" if not added_label_v_heatmap else None
        #             ax_heatmap_v_det.axvline(x=t_event, color='cyan', linestyle=':', alpha=0.7, linewidth=1.2, label=label)
        #             added_label_v_heatmap = True
        #         if added_label_v_heatmap: ax_heatmap_v_det.legend(fontsize='small')
        #         print(f"Добавлены аннотации RT событий на тепловую карту V(x,t) детекторов.")
        # else:
        #     print("Недостаточно данных для тепловой карты V(x,t) по детекторам.")
        # fig_heatmap_v_det.savefig(os.path.join(analysis_plot_dir, "heatmap_V_detectors_rt_annotated.png"))
        # plt.close(fig_heatmap_v_det)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 1. Q(x) - ЗАКОММЕНТИРОВАНО ПО ЗАПРОСУ ПОЛЬЗОВАТЕЛЯ
        # fig_q_x, ax_q_x = plt.subplots(figsize=(10, 6))
        # if Q_data_detectors is not None and cameras_coords is not None:
        #     mean_q_per_camera = np.nanmean(Q_data_detectors, axis=1)
        #     ax_q_x.plot(cameras_coords, mean_q_per_camera, marker='s', linestyle='-', color='g')
        #     ax_q_x.set_xlabel("Положение камеры на кольце, x (м)")
        #     ax_q_x.set_ylabel("Средняя интенсивность Q (ТС/с)")
        #     ax_q_x.set_title("Профиль средней интенсивности Q(x) (детекторы)")
        #     ax_q_x.grid(True)
        #     fig_q_x.savefig(os.path.join(analysis_plot_dir, "qx_profile_detectors.png"))
        # else:
        #     print("Нет данных Q для графика Q(x).")
        # plt.close(fig_q_x)

        # 2. rho(x) - ЗАКОММЕНТИРОВАНО ПО ЗАПРОСУ ПОЛЬЗОВАТЕЛЯ
        # fig_rho_x, ax_rho_x = plt.subplots(figsize=(10, 6))
        # if rho_data_detectors is not None and cameras_coords is not None:
        #     mean_rho_per_camera = np.nanmean(rho_data_detectors, axis=1)
        #     ax_rho_x.plot(cameras_coords, mean_rho_per_camera, marker='d', linestyle='-', color='purple')
        #     ax_rho_x.set_xlabel("Положение камеры на кольце, x (м)")
        #     ax_rho_x.set_ylabel("Средняя плотность rho (ТС/м)")
        #     ax_rho_x.set_title("Профиль средней плотности rho(x) (детекторы)")
        #     ax_rho_x.grid(True)
        #     fig_rho_x.savefig(os.path.join(analysis_plot_dir, "rhox_profile_detectors.png"))
        # else:
        #     print("Нет данных rho для графика rho(x).")
        # plt.close(fig_rho_x)

        # 3. Q(t) для первой камеры - ЗАКОММЕНТИРОВАНО ПО ЗАПРОСУ ПОЛЬЗОВАТЕЛЯ
        # fig_q_t, ax_q_t = plt.subplots(figsize=(10, 6))
        # if Q_data_detectors is not None and Ts_detector is not None:
        #     cam_idx_for_qt_plot = 0
        #     if Q_data_detectors.shape[0] > cam_idx_for_qt_plot:
        #         time_axis_qt = np.arange(Q_data_detectors.shape[1]) * Ts_detector
        #         ax_q_t.plot(time_axis_qt, Q_data_detectors[cam_idx_for_qt_plot, :], linestyle='-', color='g')
        #         ax_q_t.set_xlabel(f"Время t (с) (Ts={Ts_detector:.2f}c)")
        #         ax_q_t.set_ylabel(f"Интенсивность Q на камере {cam_idx_for_qt_plot} (ТС/с)")
        #         ax_q_t.set_title(f"Динамика интенсивности Q(t) на камере {cam_idx_for_qt_plot} (детекторы)")
        #         ax_q_t.grid(True)
        #         # Аннотации RT событий для Q(t) - убираем, т.к. график Q(t) уходит
        #         # if rt_wave_event_times:
        #         #     for t_event in rt_wave_event_times:
        #         #         ax_q_t.axvline(x=t_event, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        #         fig_q_t.savefig(os.path.join(analysis_plot_dir, "qt_profile_detectors.png"))
        #     else:
        #         print(f"Недостаточно камер для Q(t) для камеры {cam_idx_for_qt_plot}.")
        # else:
        #     print("Нет данных Q или Ts для графика Q(t).")
        # plt.close(fig_q_t)

        # 4. rho(t) для первой камеры - ОСТАВЛЯЕМ
        fig_rho_t, ax_rho_t = plt.subplots(figsize=(10, 6))
        if rho_data_detectors is not None and Ts_detector is not None:
            cam_idx_for_rhot_plot = 0
            if rho_data_detectors.shape[0] > cam_idx_for_rhot_plot:
                time_axis_rhot = np.arange(rho_data_detectors.shape[1]) * Ts_detector
                ax_rho_t.plot(time_axis_rhot, rho_data_detectors[cam_idx_for_rhot_plot, :], linestyle='-', color='purple')
                ax_rho_t.set_xlabel(f"Время t (с) (Ts={Ts_detector:.2f}c)")
                ax_rho_t.set_ylabel(f"Плотность rho на камере {cam_idx_for_rhot_plot} (ТС/м)")
                ax_rho_t.set_title(f"Динамика плотности rho(t) на камере {cam_idx_for_rhot_plot} (детекторы)")
                ax_rho_t.grid(True)
                
                # +++ Добавление вертикальных линий rt_wave_event_times на график rho(t) +++
                if rt_wave_event_times:
                    added_label_rho_t = False
                    for t_event in rt_wave_event_times:
                        label = "RT Wave Event" if not added_label_rho_t else None
                        ax_rho_t.axvline(x=t_event, color='gray', linestyle='--', alpha=0.7, linewidth=1.2, label=label)
                        added_label_rho_t = True
                    if added_label_rho_t: ax_rho_t.legend(fontsize='small')
                    print(f"Добавлены аннотации RT событий на график rho(t).")
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                fig_rho_t.savefig(os.path.join(analysis_plot_dir, "rhot_profile_detectors_annotated.png")) # Изменяем имя файла
            else:
                print(f"Недостаточно камер для rho(t) для камеры {cam_idx_for_rhot_plot}.")
        else:
            print("Нет данных rho или Ts для графика rho(t).")
        plt.close(fig_rho_t)

        # Старый код для добавления аннотаций на Q(t) и rho(t) был здесь, теперь только rho(t) имеет аннотации выше
        # if rt_wave_event_times:
            # ax_q_t уже не существует здесь, если Q(t) закомментирован
            # for t_event in rt_wave_event_times:
                # ax_q_t.axvline(x=t_event, color='gray', linestyle='--', alpha=0.5, linewidth=1) 
                # ax_rho_t.axvline(x=t_event, color='gray', linestyle='--', alpha=0.5, linewidth=1) # Это уже сделано выше
            # print(f"Добавлены аннотации RT событий на графики Q(t) и rho(t).")
        
        # Удаляем fig_q_x.savefig, т.к. сам график закомментирован
        # fig_q_x.savefig(os.path.join(analysis_plot_dir, "qx_plot_detectors.png"))
        # plt.close(fig_q_x)

        # 5. Heatmap Q(x,t) - ЗАКОММЕНТИРОВАНО ПО ЗАПРОСУ ПОЛЬЗОВАТЕЛЯ
        # fig_heatmap_q, ax_heatmap_q = plt.subplots(figsize=(12, 8))
        # if Q_data_detectors is not None and Q_data_detectors.size > 0 and cameras_coords is not None and Ts_detector is not None:
        #     time_steps_heatmap = Q_data_detectors.shape[1]
        #     time_axis_heatmap = np.arange(time_steps_heatmap + 1) * Ts_detector 
        #     pcm_heatmap_q = ax_heatmap_q.pcolormesh(time_axis_heatmap, cameras_coords, Q_data_detectors, 
        #                                           shading='auto', cmap='viridis', 
        #                                           vmin=np.nanmin(Q_data_detectors), vmax=np.nanmax(Q_data_detectors))
        #     fig_heatmap_q.colorbar(pcm_heatmap_q, ax=ax_heatmap_q, label='Интенсивность Q (ТС/с)')
        #     ax_heatmap_q.set_xlabel(f"Время t (с) (Ts={Ts_detector:.2f}c)")
        #     ax_heatmap_q.set_ylabel("Положение камеры на кольце, x (м)")
        #     ax_heatmap_q.set_title("x-t Тепловая карта интенсивности Q(x,t) (детекторы)")
        #     fig_heatmap_q.savefig(os.path.join(analysis_plot_dir, "heatmap_Q_detectors.png"))
        # else:
        #     print("Нет данных для тепловой карты Q(x,t).")
        # plt.close(fig_heatmap_q)

        # 6. Heatmap rho(x,t) - ЗАКОММЕНТИРОВАНО ПО ЗАПРОСУ ПОЛЬЗОВАТЕЛЯ
        # fig_heatmap_rho, ax_heatmap_rho = plt.subplots(figsize=(12, 8))
        # if rho_data_detectors is not None and rho_data_detectors.size > 0 and cameras_coords is not None and Ts_detector is not None:
        #     time_steps_heatmap_rho = rho_data_detectors.shape[1]
        #     time_axis_heatmap_rho = np.arange(time_steps_heatmap_rho + 1) * Ts_detector 
        #     pcm_heatmap_rho = ax_heatmap_rho.pcolormesh(time_axis_heatmap_rho, cameras_coords, rho_data_detectors, 
        #                                               shading='auto', cmap='cividis', 
        #                                               vmin=np.nanmin(rho_data_detectors), vmax=np.nanmax(rho_data_detectors))
        #     fig_heatmap_rho.colorbar(pcm_heatmap_rho, ax=ax_heatmap_rho, label='Плотность rho (ТС/м)')
        #     ax_heatmap_rho.set_xlabel(f"Время t (с) (Ts={Ts_detector:.2f}c)")
        #     ax_heatmap_rho.set_ylabel("Положение камеры на кольце, x (м)")
        #     ax_heatmap_rho.set_title("x-t Тепловая карта плотности rho(x,t) (детекторы)")
        #     fig_heatmap_rho.savefig(os.path.join(analysis_plot_dir, "heatmap_rho_detectors.png"))
        # else:
        #     print("Нет данных для тепловой карты rho(x,t).")
        # plt.close(fig_heatmap_rho)
        print(f"График rho(t) сохранен в {analysis_plot_dir}") # Обновляем сообщение

    else: # if not detector_data_loaded:
        print("Данные детекторов не были загружены. Пропуск анализа stop-and-go волн и построения графиков V,Q,rho по ним.")

    # --- Существующий анализ на основе FCD (если df есть) ---
    if df is not None and not df.empty : # Добавим not df.empty для надежности
        # Пример вызова plot_spacetime_heatmap с передачей rt_wave_event_times
        fig_heatmap_fcd, ax_heatmap_fcd = plt.subplots(figsize=(12, 8))
        # plot_spacetime_heatmap использует df, который уже отфильтрован по warmup_time
        plot_spacetime_heatmap(df, analysis_plot_dir, L=L, ax=ax_heatmap_fcd) 
        
        # +++ Добавление вертикальных линий rt_wave_event_times на FCD тепловую карту +++
        if rt_wave_event_times and ax_heatmap_fcd:
            added_label = False
            events_plotted_count = 0
            for t_event in rt_wave_event_times:
                if t_event >= warmup_time: # <--- ФИЛЬТРУЕМ СОБЫТИЯ ПО WARMUP_TIME
                    label = "RT Wave Event" if not added_label else None
                    ax_heatmap_fcd.axvline(x=t_event, color='white', linestyle='--', alpha=0.6, linewidth=1.2, label=label)
                    if label: # Устанавливаем added_label = True только если метка действительно была добавлена
                        added_label = True
                    events_plotted_count +=1
            
            if added_label: # Добавляем легенду только если хотя бы одна метка была нарисована (и имела label)
                ax_heatmap_fcd.legend(fontsize='small')
            
            if events_plotted_count > 0:
                print(f"Добавлено {events_plotted_count} аннотаций RT событий (для t >= {warmup_time}s) на FCD тепловую карту.")
            else:
                print(f"Нет RT событий для аннотации на FCD тепловой карте (для t >= {warmup_time}s).")
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        fig_heatmap_fcd.savefig(os.path.join(analysis_plot_dir, "spacetime_heatmap_fcd_rt_annotated.png"))
        plt.close(fig_heatmap_fcd)

        # ... (остальные существующие операции, например, FFT, если он также использует свой ax)
        # plot_fft_analysis(df, analysis_plot_dir) # Например

    # --- Сохранение сводки ---
    summary_data = {
        'data_file': os.path.basename(fcd_csv_file_path),
        'ring_length_used': L if L is not None else (df['distance'].max() if not df.empty and 'distance' in df.columns else 0),
        'mean_speed_std_dev': mean_speed_std_dev,
        'mean_speed_overall': mean_speed_overall,  # Добавляем среднюю скорость
        'waves_observed_threshold': wave_threshold,
        # Преобразуем numpy.bool_ в стандартный bool для JSON
        'waves_observed': bool(waves_observed),
        'analysis_timestamp': datetime.now().isoformat()
        # Дополнительные параметры (T, v_e, s_e_net, L) нужно будет добавить извне, 
        # когда этот скрипт вызывается из другого скрипта, который их знает.
    }
    save_analysis_summary(summary_data, analysis_plot_dir)

    print(f"Анализ завершен. Результаты сохранены в директории: {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description='Анализ FCD данных и данных детекторов для кольцевой трассы SUMO.')
    # Старый аргумент --file
    # parser.add_argument('--file', type=str, required=True, help='Путь к CSV файлу с FCD данными.')
    parser.add_argument('--results-dir', type=str, required=True, 
                        help='Путь к директории с результатами симуляции (содержит FCD CSV и данные детекторов).')
    parser.add_argument('--length', '-L', type=float, default=None, 
                        help='Длина кольцевой трассы (м). Если не задано, используется max(distance) из FCD или длина по умолчанию.')
    parser.add_argument('--warmup-time', type=float, default=150.0, 
                        help='Время прогрева симуляции (с), которое нужно отбросить в начале FCD данных.')
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Ошибка: Директория результатов {args.results_dir} не найдена.")
        return
    
    # Передаем results_dir в analyze_circle_data
    analyze_circle_data(args.results_dir, L=args.length, warmup_time=args.warmup_time)

if __name__ == "__main__":
    main() 