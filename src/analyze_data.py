import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime
import numpy as np

def analyze_data(file_path):
    """Анализирует данные из файла и создает визуализации"""
    # Создаем директорию для результатов, если её нет
    results_dir = os.path.join("..", "results", "analysis")
    os.makedirs(results_dir, exist_ok=True)
    
    # Создаем поддиректорию с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(results_dir, f"analysis_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Читаем данные
    df = pd.read_csv(file_path)
    
    # Вычисляем пройденное расстояние для каждого автомобиля
    vehicles = df['vehicle_id'].unique()
    df['distance'] = 0.0  # Добавляем колонку для расстояния
    
    # Находим начальные позиции для каждого автомобиля
    initial_positions = {}
    for vehicle in vehicles:
        vehicle_data = df[df['vehicle_id'] == vehicle]
        initial_positions[vehicle] = {
            'x': vehicle_data['x'].iloc[0],
            'y': vehicle_data['y'].iloc[0]
        }
    
    # Вычисляем расстояния относительно начальной позиции первого автомобиля
    reference_x = initial_positions[vehicles[0]]['x']
    reference_y = initial_positions[vehicles[0]]['y']
    
    for vehicle in vehicles:
        mask = df['vehicle_id'] == vehicle
        x = df.loc[mask, 'x'].values
        y = df.loc[mask, 'y'].values
        
        # Вычисляем пройденное расстояние как кумулятивную сумму перемещений
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        cumulative_distance = np.concatenate(([0], np.cumsum(distances)))
        
        # Добавляем начальное смещение относительно первого автомобиля
        initial_offset = np.sqrt(
            (initial_positions[vehicle]['x'] - reference_x)**2 + 
            (initial_positions[vehicle]['y'] - reference_y)**2
        )
        
        df.loc[mask, 'distance'] = cumulative_distance + initial_offset
    
    # График V(t)
    plt.figure(figsize=(12, 6))
    braking_time = 10.0  # Время торможения
    
    # Сначала рисуем все автомобили кроме тормозящего
    for vehicle in vehicles[1:]:
        vehicle_data = df[df['vehicle_id'] == vehicle]
        plt.plot(vehicle_data['time'], vehicle_data['speed'], 'b-', alpha=0.5)
    
    # Затем рисуем тормозящий автомобиль поверх остальных
    vehicle_data = df[df['vehicle_id'] == vehicles[0]]
    plt.plot(vehicle_data['time'], vehicle_data['speed'], 'r-', linewidth=2)
    
    # Отмечаем момент торможения
    plt.axvline(x=braking_time, color='k', linestyle='--', alpha=0.5)
    plt.text(braking_time, plt.ylim()[1], 'Торможение', rotation=90, verticalalignment='top')
    
    plt.title('Скорость от времени V(t)')
    plt.xlabel('Время (с)')
    plt.ylabel('Скорость (м/с)')
    plt.grid(True)
    plt.savefig(os.path.join(analysis_dir, 'velocity_time.png'))
    plt.close()
    
    # График S(t)
    plt.figure(figsize=(12, 6))
    
    # Сначала рисуем все автомобили кроме тормозящего
    for vehicle in vehicles[1:]:
        vehicle_data = df[df['vehicle_id'] == vehicle]
        plt.plot(vehicle_data['time'], vehicle_data['distance'], 'b-', alpha=0.5)
    
    # Затем рисуем тормозящий автомобиль поверх остальных
    vehicle_data = df[df['vehicle_id'] == vehicles[0]]
    plt.plot(vehicle_data['time'], vehicle_data['distance'], 'r-', linewidth=2)
    
    # Отмечаем момент торможения
    plt.axvline(x=braking_time, color='k', linestyle='--', alpha=0.5)
    plt.text(braking_time, plt.ylim()[1], 'Торможение', rotation=90, verticalalignment='top')
    
    plt.title('Пройденное расстояние от времени S(t)')
    plt.xlabel('Время (с)')
    plt.ylabel('Расстояние (м)')
    plt.grid(True)
    plt.savefig(os.path.join(analysis_dir, 'distance_time.png'))
    plt.close()
    
    
    # 5. Статистический анализ
    # Общая статистика
    stats = df[['speed', 'x', 'y']].describe()
    stats.to_csv(os.path.join(analysis_dir, 'statistics.csv'))
    
    # Статистика по каждому транспортному средству
    vehicle_stats = []
    for vehicle in vehicles:
        vehicle_data = df[df['vehicle_id'] == vehicle]
        stats = vehicle_data[['speed', 'x', 'y']].describe()
        stats['vehicle_id'] = vehicle
        vehicle_stats.append(stats)
    
    # Объединяем статистику всех транспортных средств
    all_vehicle_stats = pd.concat(vehicle_stats, keys=[f'ТС {v}' for v in vehicles])
    all_vehicle_stats.to_csv(os.path.join(analysis_dir, 'vehicles_statistics.csv'))
    
    print(f"Анализ завершен. Результаты сохранены в директории: {analysis_dir}")

def main():
    parser = argparse.ArgumentParser(description='Анализ данных симуляции')
    parser.add_argument('--file', type=str, required=True, help='Путь к файлу с данными для анализа')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Ошибка: Файл {args.file} не найден")
        return
    
    analyze_data(args.file)

if __name__ == "__main__":
    main() 