import os
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime

def get_edges_from_net(net_file):
    """Извлекает список ребер из net.xml файла"""
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    edges = []
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        if edge_id and not edge_id.startswith(':'):  # Пропускаем внутренние ребра
            edges.append(edge_id)
    
    return edges

def generate_straight_rou(N, eidm_params, q_fixed):
    """
    Генерирует файл маршрутов для прямой дороги.
    N - число машин
    eidm_params - параметры EIDM
    q_fixed - поток (vehicles/m/s)
    """
    # Создаем директорию для маршрутов, если её нет
    routes_dir = os.path.join("config", "routes")
    os.makedirs(routes_dir, exist_ok=True)
    
    # Получаем список ребер из net.xml
    net_file = os.path.join("config", "network", "straight.net.xml")
    edges = get_edges_from_net(net_file)
    
    # Создаем XML файл
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(routes_dir, f"straight.rou.xml")
    
    # Генерируем параметры EIDM с нормальным распределением
    accel = np.random.normal(eidm_params['accel_mean'], eidm_params['accel_std'])
    decel = np.random.normal(eidm_params['decel_mean'], eidm_params['decel_std'])
    sigma = np.random.normal(eidm_params['sigma_mean'], eidm_params['sigma_std'])
    tau = np.random.normal(eidm_params['tau_mean'], eidm_params['tau_std'])
    delta = np.random.normal(eidm_params['delta_mean'], eidm_params['delta_std'])
    stepping = np.random.normal(eidm_params['stepping_mean'], eidm_params['stepping_std'])
    
    # Генерируем параметры транспортного средства
    length = np.random.normal(eidm_params['length_mean'], eidm_params['length_std'])
    min_gap = np.random.normal(eidm_params['min_gap_mean'], eidm_params['min_gap_std'])
    max_speed = np.random.normal(eidm_params['max_speed_mean'], eidm_params['max_speed_std'])
    
    # Вычисляем плотность из потока и скорости: rho = q / v
    v = max_speed if max_speed > 0 else 1.0  # чтобы не делить на 0
    rho = q_fixed / v
    distance_between_cars = v / q_fixed  # = 1/rho
    
    with open(output_file, 'w') as f:
        # Записываем заголовок
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        # Определяем тип транспортного средства
        f.write('    <vType id="car" accel="{:.2f}" decel="{:.2f}" sigma="{:.2f}" length="{:.2f}" minGap="{:.2f}" maxSpeed="{:.2f}" speedFactor="1.0" speedDev="0.0" guiShape="passenger" carFollowModel="EIDM" tau="{:.2f}" delta="{:.2f}" stepping="{:.2f}"/>\n'.format(
            accel, decel, sigma, length, min_gap, max_speed, tau, delta, stepping))
        
        # Создаем маршрут через все ребра
        route_edges_str = ' '.join(edges)
        f.write(f'    <route id="route_0" edges="{route_edges_str}"/>\n')
        
        # Добавляем транспортные средства с заданным потоком
        for i in range(N):
            vehicle_id = f"car_{i}"
            depart_pos = i * distance_between_cars  # Начальная позиция с учетом потока
            f.write(f'    <vehicle id="{vehicle_id}" type="car" route="route_0" depart="0" departPos="{depart_pos}" insertionChecks="none" departSpeed="max" departLane="best"/>\n')
        
        f.write('</routes>')
    
    return output_file

if __name__ == "__main__":
    N = 100  # Общее количество машин
    q_fixed = 0.1  # Поток (vehicles/m/s)
    eidm_params = {
        'accel_mean': 1.8,
        'accel_std': 0.3,
        'decel_mean': 3.5,
        'decel_std': 0.3,
        'sigma_mean': 0.3,
        'sigma_std': 0.1,
        'tau_mean': 1.2,
        'tau_std': 0.1,
        'delta_mean': 4.0,
        'delta_std': 0.1,
        'stepping_mean': 0.25,
        'stepping_std': 0.1,
        'length_mean': 5.0,
        'length_std': 0.5,
        'min_gap_mean': 2,
        'min_gap_std': 0,
        'max_speed_mean': 20,  # 50 км/ч
        'max_speed_std': 0
    }
    output_file = generate_straight_rou(N, eidm_params, q_fixed)
    print(f"Сгенерирован файл маршрутов: {output_file}") 