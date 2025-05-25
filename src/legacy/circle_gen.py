import os
import sys
import traci
import sumolib
import time

# Создаем сеть из файлов
netconvert_cmd = [
    "netconvert",
    "--node-files=circle.nod.xml",
    "--edge-files=circle.edg.xml",
    "--output-file=circle.net.xml",
    "--junctions.corner-detail=20"
]

os.system(" ".join(netconvert_cmd))

# Настройки запуска SUMO
sumoBinary = "sumo-gui"
sumoConfig = "circle.sumocfg"
step_length = 0.1
sumoCmd = [sumoBinary, "-c", sumoConfig, "--step-length", str(step_length)]

try:
    traci.start(sumoCmd)
except Exception as e:
    sys.exit(1)

# Основной цикл симуляции
step = 0
try:
    while step < 3600:
        traci.simulationStep()
        
        # Получаем список всех транспортных средств
        vehicle_ids = traci.vehicle.getIDList()
        
        # Проверяем каждое транспортное средство
        for vehicle_id in vehicle_ids:
            try:
                # Получаем текущую позицию
                position = traci.vehicle.getPosition(vehicle_id)
                # Получаем текущую скорость
                speed = traci.vehicle.getSpeed(vehicle_id)
                # Получаем текущий маршрут
                route = traci.vehicle.getRoute(vehicle_id)
            except Exception:
                pass
        
        step += 1
        time.sleep(0.1)  # Добавляем небольшую задержку для стабильности

except Exception:
    pass
finally:
    traci.close()