import os
import sys
import traci
import time
import numpy as np
import pandas as pd
from datetime import datetime
from generate_circle_rou_new import generate_circle_rou
from analyze_data import analyze_data

def run_simulation(N, eidm_params):
    # Создаем папку для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("..", "data", "simulations", f"results_N{N}_accel{eidm_params['accel_mean']}_decel{eidm_params['decel_mean']}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Генерируем файл маршрутов
    generate_circle_rou(N, eidm_params)
    
    # Настройки запуска SUMO
    sumoBinary = "sumo-gui"
    sumoConfig = os.path.join("..", "config", "network", "circle.sumocfg")
    step_length = 0.1
    sumoCmd = [sumoBinary, "-c", sumoConfig, "--step-length", str(step_length)]

    # Подготовка массивов для хранения данных
    times = []
    positions = []
    speeds = []
    vehicle_ids_history = []

    try:
        # Запускаем SUMO
        traci.start(sumoCmd)
        
        # Основной цикл симуляции
        step = 0
        max_steps = 20000  # 100 секунд при step_length = 0.1
        
        while step < max_steps:
            traci.simulationStep()
            current_time = step * step_length
            
            # Получаем список всех транспортных средств
            vehicle_ids = traci.vehicle.getIDList()
            
            # Сохраняем данные для каждого транспортного средства
            for vehicle_id in vehicle_ids:
                try:
                    position = traci.vehicle.getPosition(vehicle_id)
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    
                    # Сохраняем данные
                    times.append(current_time)
                    positions.append(position)
                    speeds.append(speed)
                    vehicle_ids_history.append(vehicle_id)
                    
                    # На 100-й секунде снижаем скорость первого автомобиля
                    if current_time == 10.0 and vehicle_id == vehicle_ids[0]:
                        traci.vehicle.setSpeed(vehicle_id, 2.0)
                    # Возвращаем нормальную скорость через 10 секунду
                    elif current_time == 11.0 and vehicle_id == vehicle_ids[0]:
                        traci.vehicle.setSpeed(vehicle_id, -1)  # -1 означает максимальную скорость
                        
                except Exception as e:
                    print(f"Ошибка при получении данных для ТС {vehicle_id}: {e}")
            
            step += 1

    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        traci.close()
        
        # Создаем DataFrame с собранными данными
        df = pd.DataFrame({
            'time': times,
            'vehicle_id': vehicle_ids_history,
            'x': [pos[0] for pos in positions],
            'y': [pos[1] for pos in positions],
            'speed': speeds
        })
        
        # Сохраняем данные в CSV файл
        output_file = os.path.join(results_dir, 'simulation_data.csv')
        df.to_csv(output_file, index=False)
        
        print(f"Эксперимент завершён. Данные сохранены в {output_file}")
        return results_dir  # Возвращаем путь к папке с результатами

if __name__ == "__main__":
    # Создаем основные директории
    os.makedirs("data/simulations", exist_ok=True)
    
    # Параметры EIDM
    eidm_params = {
        'accel_mean': 2.6,
        'accel_std': 0,
        'decel_mean': 4.5,
        'decel_std': 0,
        'sigma_mean': 0.5,
        'sigma_std': 0,
        'tau_mean': 0.5,
        'tau_std': 0,
        'delta_mean': 0.5,
        'delta_std': 0,
        'stepping_mean': 0.5,
        'stepping_std': 0,
        'length_mean': 5.0,
        'length_std': 0.5,
        'min_gap_mean': 2.5,
        'min_gap_std': 0.5,
        'max_speed_mean': 22,
        'max_speed_std': 3.0
    }
    
    N = 5  # Количество машин на каждом edge
    results_dir = run_simulation(N, eidm_params)
    print(f"Результаты сохранены в директории: {results_dir}") 
    # Анализируем полученные данные
    analyze_data(os.path.join(results_dir, 'simulation_data.csv'))