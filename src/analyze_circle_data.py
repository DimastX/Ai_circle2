import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime

def analyze_circle_data(data_file):
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
        # 'vehicle_pos': 'lane_position', # Расстояние по полосе, может быть полезно
        # 'vehicle_lane': 'lane_id'
    }
    cols_to_rename = {k: v for k, v in required_cols.items() if k in df.columns}
    df.rename(columns=cols_to_rename, inplace=True)

    # Проверяем, что все необходимые колонки теперь существуют
    if not all(col in df.columns for col in ['time', 'vehicle_id', 'x', 'y', 'speed']):
        print("Ошибка: В CSV файле отсутствуют необходимые колонки (time, vehicle_id, x, y, speed) после переименования.")
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

    for vehicle in vehicles:
        mask = df['vehicle_id'] == vehicle
        vehicle_data = df[mask]
        if len(vehicle_data) > 1:
            # Координаты должны быть отсортированы по времени, что мы сделали выше
            x_coords = vehicle_data['x'].values
            y_coords = vehicle_data['y'].values
            
            dx = np.diff(x_coords)
            dy = np.diff(y_coords)
            
            distances = np.sqrt(dx**2 + dy**2)
            cumulative_distance = np.concatenate(([0], np.cumsum(distances)))
            df.loc[mask, 'distance'] = cumulative_distance
        elif len(vehicle_data) == 1:
            df.loc[mask, 'distance'] = 0.0 # Одна точка - нулевое расстояние

    plots_dir = analysis_dir # Графики сохраняем в созданную директорию анализа

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

    print(f"Анализ завершен. Графики сохранены в директории: {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description='Анализ данных симуляции кругового движения.')
    parser.add_argument('--file', type=str, required=True, help='Путь к CSV файлу с данными для анализа.')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Ошибка: Файл {args.file} не найден.")
        return
    
    analyze_circle_data(args.file)

if __name__ == "__main__":
    main() 