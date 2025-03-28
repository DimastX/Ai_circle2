import os
import sys
import traci
import time
import numpy as np
import pandas as pd
from datetime import datetime
from generate_straight_rou import generate_straight_rou
from analyze_straight_data import analyze_straight_data

def run_simulation(N, eidm_params, q_fixed, step_length=0.5, v_e=None):
    # Создаем папку для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("data", "simulations", f"straight_results_N{N}_q{q_fixed}_accel{eidm_params['accel_mean']}_decel{eidm_params['decel_mean']}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Генерация rou.xml теперь только вне этой функции!
    # generate_straight_rou(N, eidm_params, q_fixed)
    
    # Настройки запуска SUMO
    sumoBinary = "C:\\Program Files (x86)\\Eclipse\\sumo-1.22.0\\bin\\sumo.exe"
    sumoConfig = os.path.join("config", "network", "straight.sumocfg")
    sumoCmd = [sumoBinary, "-c", sumoConfig, "--step-length", str(step_length)]

    # Подготовка массивов для хранения данных
    times = []
    positions = []
    speeds = []
    vehicle_ids_history = []
    densities = []
    flows = []
    mean_speeds = []
    braking_flags = []
    times_density = []
    column_lengths = []  # Для хранения длины колонны
    times_column = []

    try:
        # Запускаем SUMO
        traci.start(sumoCmd)
        #time.sleep(2)  # Даем время на запуск SUMO
        
        # Основной цикл симуляции
        step = 0
        max_steps = 5000  # Увеличиваем время симуляции
        
        # Получаем длину ребра из конфигурации сети
        edge_length = 20000.0  # Увеличиваем длину дороги до 20 км
        
        # Флаг для отслеживания замедления 99-й машины
        ninety_ninth_car_slowed = False
        ninety_ninth_car_restored = False
        braking_time = None  # Время начала торможения
        
        simulation_active = True

        while step < max_steps and simulation_active:
            # --- Устанавливаем скорость car_0 равной v_e ---
            if v_e is not None:
                try:
                    traci.vehicle.setSpeed("car_109", v_e)
                except Exception as e:
                    print(f"Ошибка при установке скорости car_99: {e}")
                    break
            traci.simulationStep()
            current_time = step * step_length
            vehicle_ids = traci.vehicle.getIDList()
            
            # Если машин меньше двух — завершаем симуляцию (но данные за этот шаг ещё сохраняем)
            if len(vehicle_ids) < 2:
                print(f"На дороге осталось {len(vehicle_ids)} машин(ы) на шаге {step}, время {current_time} сек. Останавливаем симуляцию.")
                break
            # for vid in traci.vehicle.getIDList():
            #     print(vid, traci.vehicle.getSpeed(vid), traci.vehicle.getDistance(vid))
            
            # Получаем список всех транспортных средств
            vehicle_ids = traci.vehicle.getIDList()
            # a98 = float(traci.vehicle.getAcceleration('car_108'))
            # print(f"car_108: {a98}")
            # v98 = float(traci.vehicle.getSpeed('car_108'))
            # print(f"car_108 speed: {v98}")
            # d98 = float(traci.vehicle.getPosition('car_108')[0])
            # print(f"car_108 distance: {d98}")
            # d99 = float(traci.vehicle.getPosition('car_109')[0])
            # print(f"car_109 distance: {d99}")
            # v99 = float(traci.vehicle.getSpeed('car_109'))
            # print(f"car_109 speed: {v99}")
            # s_star = eidm_params['min_gap_mean'] + v98 * eidm_params['tau_mean'] + v98 * (v99 - v98) / (2 * np.sqrt(eidm_params['accel_mean'] * eidm_params['decel_mean']))
            # a = 1 - (v98 / eidm_params['max_speed_mean'])**4 - ((s_star)/ (d99 - d98 - eidm_params['length_mean']))**2
            # print(f"a: {a}")
            # print(f"car_109: {traci.vehicle.getAcceleration('car_109')}")
            # print("--------------------------------")

            # Проверяем время для торможения 99-й машины
            if "car_99" in vehicle_ids and not ninety_ninth_car_slowed and current_time >= 0.5:
                traci.vehicle.setAcceleration("car_99", -2.6, 2)  # Устанавливаем целевую скорость 0 для резкого торможения
                ninety_ninth_car_slowed = True
                braking_time = current_time  # Сохраняем время начала торможения
                print(f"Машина car_99 начала торможение в момент времени {braking_time} секунд")
            # Возвращаем нормальную скорость через 10 секунд
            elif ninety_ninth_car_slowed and current_time >= braking_time + 2.0:
                # traci.vehicle.setSpeed("car_99", v_e)  # -1 означает максимальную скорость
                ninety_ninth_car_restored = True
                # print(f"Машина car_99 восстановила скорость в момент времени {current_time} секунд")
            # Вычисляем плотность и поток для каждого ребра
            for edge in traci.edge.getIDList():
                if not edge.startswith(':'):  # Пропускаем внутренние ребра
                    vehicles_on_edge = traci.edge.getLastStepVehicleNumber(edge)
                    density = vehicles_on_edge / edge_length
                    flow = vehicles_on_edge / step_length
                    densities.append(density)
                    flows.append(flow)
                    times_density.append(current_time)  # Добавляем время
                    
                    # Вычисляем среднюю скорость на ребре
                    if vehicles_on_edge > 0:
                        mean_speed = traci.edge.getLastStepMeanSpeed(edge)
                    else:
                        mean_speed = 13.89  # Средняя скорость при отсутствии машин
                    mean_speeds.append(mean_speed)
            
            # Сохраняем данные для каждого транспортного средства
            reached_end = False
            x_list = []
            for vehicle_id in vehicle_ids:
                try:
                    position = traci.vehicle.getPosition(vehicle_id)
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    
                    # Добавляем флаг торможения для 99-й машины
                    is_braking = 1 if (vehicle_id == "car_99" and current_time == braking_time) else 0
                    
                    # Сохраняем данные
                    times.append(current_time)
                    positions.append(position)
                    speeds.append(speed)
                    vehicle_ids_history.append(vehicle_id)
                    braking_flags.append(is_braking)  # Добавляем флаг торможения
                    
                    # Проверяем, доехала ли машина до конца дороги
                    if position[0] >= edge_length:
                        reached_end = True
                    x_list.append(position[0])
                except Exception as e:
                    print(f"Ошибка при получении данных для ТС {vehicle_id}: {e}")
            
            if reached_end:
                print(f"Машина достигла конца дороги на шаге {step}, время {current_time} сек. Останавливаем симуляцию.")
                break
            else:
                step += 1
            
            # --- Расчет длины колонны ---
            if len(x_list) >= 2:
                length_mean = eidm_params.get('length_mean', 5.0)
                x_nose_first = max(x_list) + length_mean/2
                x_tail_last = min(x_list) - length_mean/2
                column_length = x_nose_first - x_tail_last
                column_lengths.append(column_length)
                times_column.append(current_time)

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
            'speed': speeds,
            'braking_event': braking_flags  # Добавляем колонку с флагом торможения
        })
        
        # Сохраняем данные в CSV файл
        output_file = os.path.join(results_dir, 'simulation_data.csv')
        df.to_csv(output_file, index=False)
        
        # --- Новый расчёт плотности по координатам из simulation_data.csv ---
        df_sim = pd.DataFrame({
            'time': times,
            'vehicle_id': vehicle_ids_history,
            'x': [pos[0] for pos in positions],
            'y': [pos[1] for pos in positions],
            'speed': speeds,
            'braking_event': braking_flags
        })
        density_list = []
        flow_list = []
        mean_speed_list = []
        time_list = []
        for t, group in df_sim.groupby('time'):
            N = len(group)
            if N >= 2:
                x_max = group['x'].max()
                x_min = group['x'].min()
                density = N / (x_max - x_min) if (x_max - x_min) > 0 else np.nan
            else:
                density = np.nan
            mean_speed = group['speed'].mean() if N > 0 else np.nan
            flow = density * mean_speed if not np.isnan(density) and not np.isnan(mean_speed) else np.nan
            density_list.append(density)
            flow_list.append(flow)
            mean_speed_list.append(mean_speed)
            time_list.append(t)
        density_flow_df = pd.DataFrame({
            'time': time_list,
            'density': density_list,
            'flow': flow_list,
            'mean_speed': mean_speed_list
        })
        density_flow_file = os.path.join(results_dir, 'density_flow_data.csv')
        density_flow_df.to_csv(density_flow_file, index=False)
        print(f'density_flow_data.csv сохранён: {density_flow_file}')
        # column_length.csv больше не сохраняем
        
        print(f"Эксперимент завершён. Данные сохранены в {results_dir}")
        return results_dir

if __name__ == "__main__":
    # Создаем основные директории
    os.makedirs("data/simulations", exist_ok=True)
    
    # Параметры EIDM
    eidm_params = {
        'accel_mean': 2.6,
        'accel_std': 0,
        'decel_mean': 4.5,
        'decel_std': 0,
        'sigma_mean': 0,
        'sigma_std': 0,
        'tau_mean': 1.2,
        'tau_std': 0,
        'delta_mean': 4,
        'delta_std': 0,
        'stepping_mean': 0.5,
        'stepping_std': 0,
        'length_mean': 5.0,
        'length_std': 0,
        'min_gap_mean': 2,
        'min_gap_std': 0,
        'max_speed_mean': 20,
        'max_speed_std': 0.0
    }
    
    N = 100  # Количество машин
    q_fixed = 0.769  # Поток (vehicles/m/s)
    step_length = 0.5
    
    # Запускаем симуляцию
    results_dir = run_simulation(N, eidm_params, q_fixed, step_length=step_length)
    print(f"Результаты сохранены в директории: {results_dir}")
    # Анализируем полученные данные
    analyze_straight_data(os.path.join(results_dir, 'simulation_data.csv')) 