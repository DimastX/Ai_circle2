import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime
import numpy as np

def analyze_straight_data(data_file):
    """Анализирует данные из файла и создает визуализации"""
    # Создаем директорию для результатов, если её нет
    results_dir = os.path.join("results", "analysis")
    os.makedirs(results_dir, exist_ok=True)
    
    # Создаем поддиректорию с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(results_dir, f"straight_analysis_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Читаем данные
    df = pd.read_csv(data_file)
    
    # Создаем директорию для графиков
    plots_dir = os.path.join(os.path.dirname(data_file), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Находим момент торможения
    braking_time = df[df['braking_event'] == 1]['time'].iloc[0] if 1 in df['braking_event'].values else None
    
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
    
    # График V(t) для всех автомобилей
    plt.figure(figsize=(12, 6))
    
    # Сначала рисуем все автомобили кроме тормозящего
    for vehicle in vehicles[1:]:
        vehicle_data = df[df['vehicle_id'] == vehicle]
        plt.plot(vehicle_data['time'], vehicle_data['speed'], 'b-', alpha=0.5)
    
    # Затем рисуем тормозящий автомобиль поверх остальных
    vehicle_data = df[df['vehicle_id'] == 'car_9']
    plt.plot(vehicle_data['time'], vehicle_data['speed'], 'r-', linewidth=2)
    
    # Отмечаем момент торможения
    if braking_time is not None:
        plt.axvline(x=braking_time, color='k', linestyle='--', alpha=0.5)
        plt.text(braking_time, plt.ylim()[1], 'Торможение', rotation=90, verticalalignment='top')
    
    plt.title('Скорость от времени V(t)')
    plt.xlabel('Время (с)')
    plt.ylabel('Скорость (м/с)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'velocity_time.png'))
    plt.close()
    
    # График S(t) для всех автомобилей
    plt.figure(figsize=(12, 6))
    
    # Сначала рисуем все автомобили кроме тормозящего
    for vehicle in vehicles[1:]:
        vehicle_data = df[df['vehicle_id'] == vehicle]
        plt.plot(vehicle_data['time'], vehicle_data['distance'], 'b-', alpha=0.5)
    
    # Затем рисуем тормозящий автомобиль поверх остальных
    vehicle_data = df[df['vehicle_id'] == 'car_9']
    plt.plot(vehicle_data['time'], vehicle_data['distance'], 'r-', linewidth=2)
    
    # Отмечаем момент торможения
    if braking_time is not None:
        plt.axvline(x=braking_time, color='k', linestyle='--', alpha=0.5)
        plt.text(braking_time, plt.ylim()[1], 'Торможение', rotation=90, verticalalignment='top')
    
    plt.title('Пройденное расстояние от времени S(t)')
    plt.xlabel('Время (с)')
    plt.ylabel('Расстояние (м)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'distance_time.png'))
    plt.close()
    
    # График V(t) только для каждой десятой машины
    plt.figure(figsize=(12, 6))
    
    # Рисуем каждую десятую машину разными цветами
    colors = ['r', 'g', 'c', 'm', 'y', 'k']
    for i, vehicle in enumerate(vehicles):
        if vehicle.endswith('9'):  # Каждая десятая машина
            vehicle_data = df[df['vehicle_id'] == vehicle]
            color = colors[i % len(colors)]
            plt.plot(vehicle_data['time'], vehicle_data['speed'], f'{color}-', linewidth=2, label=f'Машина {vehicle}')
    
    # Отмечаем момент торможения
    if braking_time is not None:
        plt.axvline(x=braking_time, color='k', linestyle='--', alpha=0.5)
        plt.text(braking_time, plt.ylim()[1], 'Торможение', rotation=90, verticalalignment='top')
    
    plt.title('Скорость от времени V(t) для каждой десятой машины')
    plt.xlabel('Время (с)')
    plt.ylabel('Скорость (м/с)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'velocity_time_tenth.png'))
    plt.close()
    
    # График S(t) только для каждой десятой машины
    plt.figure(figsize=(12, 6))
    
    # Рисуем каждую десятую машину разными цветами
    for i, vehicle in enumerate(vehicles):
        if vehicle.endswith('9'):  # Каждая десятая машина
            vehicle_data = df[df['vehicle_id'] == vehicle]
            color = colors[i % len(colors)]
            plt.plot(vehicle_data['time'], vehicle_data['distance'], f'{color}-', linewidth=2, label=f'Машина {vehicle}')
    
    # Отмечаем момент торможения
    if braking_time is not None:
        plt.axvline(x=braking_time, color='k', linestyle='--', alpha=0.5)
        plt.text(braking_time, plt.ylim()[1], 'Торможение', rotation=90, verticalalignment='top')
    
    plt.title('Пройденное расстояние от времени S(t) для каждой десятой машины')
    plt.xlabel('Время (с)')
    plt.ylabel('Расстояние (м)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'distance_time_tenth.png'))
    plt.close()
    
    # График V(x) для всех автомобилей
    plt.figure(figsize=(12, 6))
    
    # Сначала рисуем все автомобили кроме тормозящего
    for vehicle in vehicles[1:]:
        vehicle_data = df[df['vehicle_id'] == vehicle]
        plt.plot(vehicle_data['distance'], vehicle_data['speed'], 'b-', alpha=0.5)
    
    # Затем рисуем тормозящий автомобиль поверх остальных
    vehicle_data = df[df['vehicle_id'] == 'car_9']
    plt.plot(vehicle_data['distance'], vehicle_data['speed'], 'r-', linewidth=2)
    
    # Отмечаем момент торможения
    if braking_time is not None:
        braking_distance = df[(df['vehicle_id'] == 'car_9') & (df['time'] == braking_time)]['distance'].iloc[0]
        plt.axvline(x=braking_distance, color='k', linestyle='--', alpha=0.5)
        plt.text(braking_distance, plt.ylim()[1], 'Торможение', rotation=90, verticalalignment='top')
    
    plt.title('Скорость от положения V(x)')
    plt.xlabel('Положение на трассе (м)')
    plt.ylabel('Скорость (м/с)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'velocity_position.png'))
    plt.close()
    
    # График V(x) только для каждой десятой машины
    plt.figure(figsize=(12, 6))
    
    # Рисуем каждую десятую машину разными цветами
    for i, vehicle in enumerate(vehicles):
        if vehicle.endswith('9'):  # Каждая десятая машина
            vehicle_data = df[df['vehicle_id'] == vehicle]
            color = colors[i % len(colors)]
            plt.plot(vehicle_data['distance'], vehicle_data['speed'], f'{color}-', linewidth=2, label=f'Машина {vehicle}')
    
    # Отмечаем момент торможения
    if braking_time is not None:
        braking_distance = df[(df['vehicle_id'] == 'car_9') & (df['time'] == braking_time)]['distance'].iloc[0]
        plt.axvline(x=braking_distance, color='k', linestyle='--', alpha=0.5)
        plt.text(braking_distance, plt.ylim()[1], 'Торможение', rotation=90, verticalalignment='top')
    
    plt.title('Скорость от положения V(x) для каждой десятой машины')
    plt.xlabel('Положение на трассе (м)')
    plt.ylabel('Скорость (м/с)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'velocity_position_tenth.png'))
    plt.close()
    
    print(f"Графики сохранены в директории: {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description='Анализ данных симуляции на прямой дороге')
    parser.add_argument('--file', type=str, required=True, help='Путь к файлу с данными для анализа')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Ошибка: Файл {args.file} не найден")
        return
    
    analyze_straight_data(args.file)

if __name__ == "__main__":
    main() 