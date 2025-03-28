import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def analyze_simulation_data(file_path):
    # Читаем данные
    df = pd.read_csv(file_path)
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Получаем список уникальных транспортных средств
    vehicles = df['vehicle_id'].unique()
    
    # Цветовая карта для разных транспортных средств
    colors = plt.cm.rainbow(np.linspace(0, 1, len(vehicles)))
    
    # Находим момент торможения (10 секунда)
    braking_time = 10.0
    
    # Строим графики для каждого транспортного средства
    for i, vehicle in enumerate(vehicles):
        vehicle_data = df[df['vehicle_id'] == vehicle]
        
        # Вычисляем пройденное расстояние
        x = vehicle_data['x'].values
        y = vehicle_data['y'].values
        positions = np.sqrt(x**2 + y**2)
        
        # График расстояния
        ax1.plot(vehicle_data['time'], positions, 
                label=f'ТС {vehicle}', color=colors[i])
        
        # График скорости
        ax2.plot(vehicle_data['time'], vehicle_data['speed'], 
                label=f'ТС {vehicle}', color=colors[i])
        
        # Отмечаем момент торможения
        if vehicle == vehicles[0]:  # Для первого транспортного средства
            ax2.axvline(x=braking_time, color='r', linestyle='--', alpha=0.5)
            ax2.axvline(x=braking_time+1, color='g', linestyle='--', alpha=0.5)
            ax2.text(braking_time, ax2.get_ylim()[1], 'Торможение', 
                    rotation=90, verticalalignment='top')
            ax2.text(braking_time+1, ax2.get_ylim()[1], 'Возврат скорости', 
                    rotation=90, verticalalignment='top')
    
    # Настраиваем графики
    ax1.set_title('Пройденное расстояние S(t)')
    ax1.set_xlabel('Время (с)')
    ax1.set_ylabel('Расстояние (м)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_title('Скорость V(t)')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Скорость (м/с)')
    ax2.grid(True)
    ax2.legend()
    
    # Добавляем информацию о торможении
    braking_info = f"Торможение: ТС {vehicles[0]} снизил скорость до 2 м/с на {braking_time} секунде"
    fig.suptitle(braking_info, y=0.95)
    
    # Создаем директорию для графиков
    plots_dir = os.path.join("data", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Сохраняем график
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f'analysis_{timestamp}.png'))
    plt.close()

if __name__ == "__main__":
    # Путь к файлу с данными
    file_path = os.path.join("data", "simulations", "results_N1_accel2.6_decel4.5_20240328_123456", "simulation_data.csv")
    analyze_simulation_data(file_path) 