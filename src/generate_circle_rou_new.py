import os
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime

def get_sorted_edges(edges):
    """Сортирует ребра в правильном порядке: 0_1, 1_2, 2_3, ..."""
    # Создаем словарь для быстрого поиска ребер по начальному узлу
    edge_dict = {}
    for edge in edges:
        start, end = edge.split('_')
        edge_dict[start] = edge
    
    # Собираем ребра в правильном порядке
    sorted_edges = []
    current_start = "0"
    
    while len(sorted_edges) < len(edges):
        if current_start in edge_dict:
            sorted_edges.append(edge_dict[current_start])
            current_start = edge_dict[current_start].split('_')[1]
        else:
            break
    
    return sorted_edges

def get_edges_from_net(net_file):
    """Извлекает список ребер из net.xml файла"""
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    edges = []
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        if edge_id and not edge_id.startswith(':'):  # Пропускаем внутренние ребра (начинающиеся с ':')
            edges.append(edge_id)
    
    return edges

def generate_circle_rou(N, eidm_params):
    # Создаем директорию для маршрутов, если её нет
    routes_dir = os.path.join("..", "config", "routes")
    os.makedirs(routes_dir, exist_ok=True)
    
    # Получаем список ребер из net.xml и сортируем их
    net_file = os.path.join("..", "config", "network", "circle.net.xml")
    edges = get_edges_from_net(net_file)
    sorted_edges = get_sorted_edges(edges)
    
    # Создаем XML файл
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(routes_dir, f"circle.rou.xml")
    
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
    
    with open(output_file, 'w') as f:
        # Записываем заголовок
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        # Определяем тип транспортного средства
        f.write('    <vType id="car" accel="{:.2f}" decel="{:.2f}" sigma="{:.2f}" length="{:.2f}" minGap="{:.2f}" maxSpeed="{:.2f}" guiShape="passenger" carFollowModel="EIDM" tau="{:.2f}" delta="{:.2f}" stepping="{:.2f}"/>\n'.format(
            accel, decel, sigma, length, min_gap, max_speed, tau, delta, stepping))
        
        # Создаем маршруты, начинающиеся с каждого ребра
        for i, start_edge in enumerate(sorted_edges):
            # Создаем последовательность ребер для маршрута
            route_edges = []
            current_idx = i
            for _ in range(len(sorted_edges)):
                route_edges.append(sorted_edges[current_idx])
                current_idx = (current_idx + 1) % len(sorted_edges)
            
            route_edges_str = ' '.join(route_edges)
            
            # Добавляем маршрут
            f.write(f'    <route id="route_{i}" edges="{route_edges_str}" repeat="100"/>\n')
            
            # Добавляем транспортные средства для этого маршрута
            for j in range(N):
                vehicle_id = f"car_{i}_{j}"
                depart_pos = j * 87.15 / N # Равномерное распределение по длине ребра (87.15 метров)
                f.write(f'    <vehicle id="{vehicle_id}" type="car" route="route_{i}" depart="0" departPos="{depart_pos}" departSpeed="max"/>\n')
        
        f.write('</routes>')
    
    return output_file

if __name__ == "__main__":
    N = 7  # Количество машин на каждом маршруте
    eidm_params = {
        'accel_mean': 2.6,
        'accel_std': 0.5,
        'decel_mean': 4.5,
        'decel_std': 0.5,
        'sigma_mean': 0.5,
        'sigma_std': 0.1,
        'tau_mean': 0.5,
        'tau_std': 0.1,
        'delta_mean': 0.5,
        'delta_std': 0.1,
        'stepping_mean': 0.5,
        'stepping_std': 0.1,
        'length_mean': 5.0,
        'length_std': 0.5,
        'min_gap_mean': 2.5,
        'min_gap_std': 0.5,
        'max_speed_mean': 13.89,
        'max_speed_std': 1.0
    }
    output_file = generate_circle_rou(N, eidm_params)
    print(f"Сгенерирован файл маршрутов: {output_file}")