import os
import subprocess
import argparse
import xml.etree.ElementTree as ET
import tempfile
import shutil
import numpy as np
from datetime import datetime
import traci
import pandas as pd
import json
import sys

# Импорт из нашего нового модуля для взаимодействия с TraCI
# Предполагаем, что traci_interaction.py находится в том же каталоге src
import traci_interaction as ti

# --- ДОБАВЛЕНО: Динамическое добавление корневой директории проекта в sys.path ---
# Это нужно, чтобы найти vsl_controller.py, который находится в корне проекта
try:
    current_script_path = os.path.abspath(__file__)
    project_root_directory = os.path.dirname(os.path.dirname(current_script_path))
    if project_root_directory not in sys.path:
        sys.path.insert(0, project_root_directory)
    from vsl_controller import VSLController # <<< ДОБАВЛЕНО
    VSL_CONTROLLER_AVAILABLE = True
except ImportError:
    print("ПРЕДУПРЕЖДЕНИЕ: Не удалось импортировать VSLController. Убедитесь, что vsl_controller.py находится в корневой директории проекта.")
    VSL_CONTROLLER_AVAILABLE = False
# --- КОНЕЦ ДОБАВЛЕННОГО БЛОКА ---

# Константы для детекторов
RING_EDGE_ID = "a"  # ID кольцевого ребра из 1k.net.xml
FIXED_RING_LENGTH = 1000  # Длина кольца, как в eidm_stability_analysis.py
DETECTOR_SPACING_M = 100.0  # Желаемое расстояние между детекторами
DETECTOR_SAMPLING_PERIOD_S = 1.0  # Период опроса детекторов (Ts)
DETECTOR_FILE_NAME = "detectors.add.xml" # Имя файла с детекторами

def generate_detector_additional_file(output_dir, edge_id, num_detectors, detector_positions, detector_ids, sampling_period):
    """Генерирует .add.xml файл с E1 детекторами."""
    additional_file_path = os.path.join(output_dir, DETECTOR_FILE_NAME)
    
    root = ET.Element("additional")
    for i in range(num_detectors):
        det_id = detector_ids[i]
        pos = detector_positions[i]
        # Атрибут 'file' для вывода данных детектора формально нужен SUMO,
        # даже если мы будем читать через TraCI. Зададим уникальное имя.
        ET.SubElement(root, "e1Detector", id=det_id, lane=f"{edge_id}_0", pos=f"{pos:.2f}", 
                      freq=f"{sampling_period:.2f}", file=f"{det_id}_output.xml", friendlyPos="true")
    
    tree = ET.ElementTree(root)
    # Для красивого форматирования XML (с отступами)
    ET.indent(tree, space="    ") 
    tree.write(additional_file_path, encoding="UTF-8", xml_declaration=True)
    print(f"Файл с детекторами сохранен: {additional_file_path}")
    return additional_file_path # Возвращаем путь к файлу

def modify_routes_file(original_rou_file, temp_rou_file, num_vehicles, tau_value=None):
    """Копирует и изменяет файл маршрутов, устанавливая количество ТС и опционально tau."""
    shutil.copyfile(original_rou_file, temp_rou_file)
    tree = ET.parse(temp_rou_file)
    root = tree.getroot()

    # Изменение количества ТС
    # Ищем flow элемент по ID '1', как в config/circles/input_routes.rou.xml
    flow_element = root.find(".//flow[@id='1']")
    if flow_element is not None:
        flow_element.set("number", str(num_vehicles))
        print(f"Модифицирован файл маршрутов {temp_rou_file} для {num_vehicles} ТС (найден flow[@id='1']).")
    else:
        raise ValueError("Элемент <flow id='1'> не найден в файле маршрутов.")

    # Изменение tau для vType (если tau_value предоставлен)
    if tau_value is not None:
        # Ищем vType с id='custom', как в config/circles/input_routes.rou.xml
        vtype_element = root.find(".//vType[@id='custom']")
        if vtype_element is not None:
            current_model = vtype_element.get("carFollowModel")
            if current_model and "IDM" in current_model: # Проверяем, что используется IDM
                vtype_element.set("tau", str(tau_value))
                print(f"Параметр tau для vType 'custom' изменен на {tau_value} в {temp_rou_file}.")
            else:
                print(f"Предупреждение: vType 'custom' (модель: {current_model}) не использует IDM или атрибут carFollowModel отсутствует. Tau не изменен.")
        else:
            print(f"Предупреждение: Элемент <vType id='custom'> не найден в {temp_rou_file}. Tau не изменен.")
    
    tree.write(temp_rou_file)

def modify_sumocfg_file(original_sumocfg_file_path, temp_sumocfg_file_dest, 
                          temp_rou_file_basename, num_vehicles, fcd_output_file_abs_path,
                          original_config_dir_abs_path, detector_add_file_basename=None):
    """Копирует и изменяет файл конфигурации SUMO."""
    shutil.copyfile(original_sumocfg_file_path, temp_sumocfg_file_dest)
    tree = ET.parse(temp_sumocfg_file_dest)
    root = tree.getroot()
    input_element = root.find(".//input")
    if input_element is None:
        raise ValueError("Элемент <input> не найден в .sumocfg.")

    # Обновляем путь к файлу маршрутов (должен быть относительным к temp_sumocfg_file_dest)
    route_files_element = input_element.find(".//route-files")
    if route_files_element is not None:
        route_files_element.set("value", temp_rou_file_basename) # temp_rou_file_basename это просто имя файла
    else:
        raise ValueError("Элемент <route-files> не найден в .sumocfg.")

    # Обновляем путь к файлу сети на абсолютный
    net_file_element = input_element.find(".//net-file")
    if net_file_element is not None:
        original_net_file_name = net_file_element.get("value")
        net_file_element.set("value", os.path.join(original_config_dir_abs_path, original_net_file_name))
    else:
        raise ValueError("Элемент <net-file> не найден в .sumocfg.")

    # Обновляем путь к дополнительным файлам на абсолютный
    additional_files_element = input_element.find(".//additional-files")
    
    # Собираем все additional файлы
    additional_files_values = []
    if additional_files_element is not None and additional_files_element.get("value"):
        original_add_files = additional_files_element.get("value").split(',')
        for fname in original_add_files:
            stripped_fname = fname.strip()
            # Проверяем, не является ли путь уже абсолютным 
            # или не является ли он путем к файлу детекторов, который мы добавляем (по имени)
            if not os.path.isabs(stripped_fname) and stripped_fname != detector_add_file_basename:
                 additional_files_values.append(os.path.join(original_config_dir_abs_path, stripped_fname))
            else:
                 additional_files_values.append(stripped_fname) # Оставляем как есть (абсолютный или наш файл)

    if detector_add_file_basename: 
        if detector_add_file_basename not in additional_files_values: # Избегаем дублирования
            additional_files_values.append(detector_add_file_basename)

    if additional_files_values:
        if additional_files_element is None:
            additional_files_element = ET.SubElement(input_element, "additional-files")
        additional_files_element.set("value", ",".join(additional_files_values))
        print(f"Дополнительные файлы в sumocfg: {additional_files_element.get('value')}")
    elif additional_files_element is not None:
        # Если изначально были additional files, но мы ничего не добавили, и их там больше нет (хотя это маловероятно)
        input_element.remove(additional_files_element)
        print("Элемент <additional-files> был пустым и удален или не существовал.")
    # else: (Если additional_files_element был None и detector_add_file_basename тоже None)
        # print("Предупреждение: элемент <additional-files> не найден и не был создан (нет файлов для добавления).")

    # Обновляем max-num-vehicles
    max_vehicles_element = root.find(".//max-num-vehicles")
    if max_vehicles_element is not None:
        max_vehicles_element.set("value", str(num_vehicles))
    else:
        print("Предупреждение: элемент <max-num-vehicles> не найден. Он не будет изменен.")

    # Удаляем настройки GUI, если они есть, чтобы избежать проблем с viewsettings.xml
    gui_only_element = root.find(".//gui_only")
    if gui_only_element is not None:
        gui_settings_file_el = gui_only_element.find(".//gui-settings-file")
        if gui_settings_file_el is not None:
            gui_only_element.remove(gui_settings_file_el)
            print("Элемент <gui-settings-file> удален из временной конфигурации.")
        # Если gui_only стал пустым после удаления, удаляем и его
        if not list(gui_only_element) and not gui_only_element.text and not len(gui_only_element.attrib):
            # Более надежный способ найти родителя и удалить: получить родительскую карту
            # Однако, ElementTree не предоставляет прямого parent.remove(child)
            # Проще всего найти родителя <configuration> и в нем удалить <gui_only>
            configuration_element = root # Предполагаем, что root это <configuration>
            try:
                configuration_element.remove(gui_only_element)
                print("Пустой элемент <gui_only> удален из временной конфигурации.")
            except ValueError: # Если gui_only_element не прямой дочерний элемент root
                # Это не должно произойти, если root это <configuration>
                print("Не удалось удалить пустой элемент <gui_only>.") 

    # Устанавливаем или обновляем FCD вывод (абсолютный путь)
    output_element = root.find(".//output")
    if output_element is None:
        output_element = ET.SubElement(root, "output")

    fcd_output_element = output_element.find(".//fcd-output")
    if fcd_output_element is None:
        fcd_output_element = ET.SubElement(output_element, "fcd-output")
    
    fcd_output_element.set("value", fcd_output_file_abs_path) # Устанавливаем имя файла
    fcd_output_element.set("distance", "true") # Явно запрашиваем вывод "distance" (одометра)

    fcd_odometer_element = fcd_output_element.find(".//fcd-output.attributes")
    if fcd_odometer_element is None:
        fcd_odometer_element = ET.SubElement(fcd_output_element, "fcd-output.attributes")
    fcd_odometer_element.set("value", "odometer,speed,x,y,pos")
    # Убедимся, что другие атрибуты не удаляются и не перезаписываются, если они были
    # Например, если fcd_output_element уже существовал с какими-то атрибутами, 
    # мы просто добавляем/обновляем value и distance.

    # 3. Настраиваем fcd-output.geo (если нужно)
    fcd_geo_element = output_element.find(".//fcd-output.geo")
    if fcd_geo_element is None:
        fcd_geo_element = ET.SubElement(output_element, "fcd-output.geo")
    fcd_geo_element.set("value", "true")
    # --- Конец изменений для FCD вывода ---

    # Устанавливаем время окончания симуляции
    time_element = root.find(".//time")
    if time_element is None:
        time_element = ET.SubElement(root, "time")
    
    end_element = time_element.find(".//end")
    if end_element is None:
        end_element = ET.SubElement(time_element, "end")
    end_element.set("value", "2000")
    print(f"Время окончания симуляции установлено на 2000с во временной конфигурации.")

    tree.write(temp_sumocfg_file_dest)
    print(f"Модифицирован файл конфигурации {temp_sumocfg_file_dest}.")


def run_simulation(sumo_binary, temp_sumocfg_file, results_dir, config_name, timestamp, 
                   detector_ids, cameras_coords, detector_sampling_period, simulation_duration_s,
                   vsl_init_params=None):
    """Запускает симуляцию SUMO с TraCI, собирает данные с детекторов, 
    и опционально управляет VSL.
    """
    
    sumo_cmd = [sumo_binary, "-c", temp_sumocfg_file, "--no-warnings", "true"]
    
    # Преобразуем detector_ids в set для быстрой проверки принадлежности
    set_detector_ids = set(detector_ids)
    
    # Данные для сохранения
    # Словарь для хранения данных: detector_id -> {time: [], speed: [], count: []}
    raw_detector_data = {det_id: {'time': [], 'meanSpeed': [], 'vehicleCount': []} for det_id in detector_ids}

    # VSL Controller Instance
    vsl_controller_instance = None
    rt_wave_detector = None # Инициализируем здесь, чтобы был доступен в finally

    try:
        print(f"Подготовка к запуску SUMO с TraCI: {' '.join(sumo_cmd)}")
        
        # Измененный вызов ti.connect_sumo
        if not ti.connect_sumo(sumo_cmd): # Передаем весь sumo_cmd
            print("Не удалось запустить SUMO или подключиться через TraCI.")
            # Убедимся, что rt_wave_detector обработан в finally, даже если здесь выход
            return False, None 

        print(f"TraCI: Успешно подключено к SUMO. Версия TraCI: {traci.getVersion()}")

        # --- ИНИЦИАЛИЗАЦИЯ VSL CONTROLLER ПОСЛЕ TRACI.CONNECT ---
        if vsl_init_params and VSL_CONTROLLER_AVAILABLE:
            try:
                # Устанавливаем актуальное TraCI соединение
                vsl_init_params_with_traci = {**vsl_init_params, "traci_conn": traci}
                vsl_controller_instance = VSLController(**vsl_init_params_with_traci)
                print("VSLController успешно инициализирован.")
            except Exception as e:
                print(f"ОШИБКА VSL: Не удалось инициализировать VSLController: {e}")
                vsl_controller_instance = None # Убедимся, что он None если инициализация не удалась
        
        # --- ИНИЦИАЛИЗАЦИЯ RealTimeWaveDetector (ТОЛЬКО ЕСЛИ VSL НЕ АКТИВЕН) ---
        if not vsl_controller_instance: # Если VSL не активен (или не удалось его инициализировать)
            print("VSL не активен, инициализируем RealTimeWaveDetector.")
            rt_wave_detector = ti.RealTimeWaveDetector(
                detector_ids_ordered=detector_ids, 
                cameras_coords=cameras_coords,
                Ts=detector_sampling_period
            )
        else:
            print("VSL активен, RealTimeWaveDetector не будет инициализирован.")

        current_sumo_time = 0.0
        simulation_step_length = traci.simulation.getDeltaT()
        step_count = 0
        max_steps = int(simulation_duration_s / simulation_step_length) if simulation_step_length > 0 else 0

        while current_sumo_time < simulation_duration_s:
            traci.simulationStep()
            current_sumo_time = traci.simulation.getTime()
            step_count += 1

            # --- СБОР ДАННЫХ И УПРАВЛЕНИЕ VSL ---
            if vsl_controller_instance:
                vsl_controller_instance.step(current_sumo_time) # VSL контроллер делает свой шаг
            
            # --- СБОР ДАННЫХ ДЛЯ RealTimeWaveDetector (ТОЛЬКО ЕСЛИ АКТИВЕН) ---
            if rt_wave_detector: # Если RTWD был инициализирован
                try:
                    # Получаем данные с детекторов
                    new_readings = ti.get_detector_data(detector_ids) # detector_ids - это аргумент run_simulation
                    # Обновляем детектор волн и выполняем обнаружение
                    rt_wave_detector.update_and_detect(current_sumo_time, new_readings)
                except AttributeError as e:
                    print(f"Ошибка атрибута при работе с RealTimeWaveDetector (возможно, метод отсутствует): {e}. RTWD может не работать корректно.")
                except Exception as e:
                    print(f"Непредвиденная ошибка при работе с RealTimeWaveDetector: {e}")

            if step_count % 100 == 0 or step_count == 1 : # Логируем каждые 100 шагов + первый шаг
                 print(f"TraCI: Собраны данные с детекторов на шаге {step_count}/{max_steps} (Время SUMO: {current_sumo_time:.2f}s)")
        
        print(f"TraCI: Целевая длительность симуляции ({simulation_duration_s}s) достигнута или превышена на времени {current_sumo_time:.2f}s. Остановка.")

    except traci.TraCIException as e:
        print(f"TraCI_Exception: Ошибка во время симуляции SUMO: {e}")
        return False, None # Симуляция не удалась
    except Exception as e:
        # Логируем ошибку + полный стектрейс
        print(f"SUMO/TraCI: Непредвиденная ошибка в цикле симуляции: {e}")
        import traceback
        traceback.print_exc() # Печатаем полный стектрейс для диагностики
        return False, None
    finally:
        # --- ЗАВЕРШЕНИЕ РАБОТЫ VSL КОНТРОЛЛЕРА ---
        if vsl_controller_instance:
            vsl_controller_instance.close_log()
            print("VSLController: Лог закрыт.")

        # --- ЗАВЕРШЕНИЕ РАБОТЫ RealTimeWaveDetector (ТОЛЬКО ЕСЛИ АКТИВЕН) ---
        if rt_wave_detector: # Если RTWD был инициализирован
            try:
                # Логика finalize_and_save_events теперь обрабатывается иначе или не требуется здесь,
                # так как get_detected_event_details() можно вызвать после цикла, если нужно.
                # rt_wave_detector.finalize_and_save_events() # Этот метод может отсутствовать
                # print("RealTimeWaveDetector: События сохранены.")
                pass # Оставляем pass, так как прямого аналога finalize_and_save_events нет, 
                     # а get_detected_event_details() вызывается по необходимости позже.
            except AttributeError:
                # print("RealTimeWaveDetector: метод finalize_and_save_events не найден. События не сохранены.")
                pass # Если бы был метод, но он отсутствовал бы
            except Exception as e:
                print(f"Ошибка при (удаленной) попытке завершения работы RealTimeWaveDetector: {e}")

        if traci.isLoaded():
            ti.close_sumo()
        print("Симуляция SUMO с TraCI успешно завершена.")

    return True, rt_wave_detector # <<< ВОЗВРАЩАЕМ rt_wave_detector

def convert_fcd_xml_to_csv(xml_file, csv_file, sumo_tools_dir):
    """Конвертирует FCD XML в CSV с помощью xml2csv.py."""
    script_path = os.path.join(sumo_tools_dir, "xml", "xml2csv.py")
    if not os.path.isfile(script_path):
        print(f"Ошибка: Скрипт {script_path} не найден или не является файлом. Укажите правильный путь к директории SUMO tools.")
        return False
        
    cmd = ["python", script_path, xml_file, "-s", ",", "-o", csv_file]
    print(f"Конвертация XML в CSV: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=120) # Таймаут 2 минуты для конвертации
        if process.returncode != 0:
            print(f"Ошибка при конвертации XML в CSV (код возврата: {process.returncode}):")
            if stdout: print(f"STDOUT (xml2csv):\n{stdout.decode(errors='replace')}")
            if stderr: print(f"STDERR (xml2csv):\n{stderr.decode(errors='replace')}")
            return False
        else:
            print(f"FCD XML успешно конвертирован в {csv_file}.")
            if stdout: print(f"STDOUT (xml2csv):\n{stdout.decode(errors='replace')}")
            if stderr: print(f"STDERR (xml2csv - может содержать инфо):\n{stderr.decode(errors='replace')}")
            return True
    except subprocess.TimeoutExpired:
        print("Ошибка: Конвертация XML в CSV превысила лимит времени.")
        process.kill()
        stdout, stderr = process.communicate()
        if stdout: print(f"STDOUT (timeout):\n{stdout.decode(errors='replace')}")
        if stderr: print(f"STDERR (timeout):\n{stderr.decode(errors='replace')}")
        return False
    except FileNotFoundError:
        print(f"Ошибка: 'python' не найден или скрипт xml2csv.py ({script_path}) не найден.")
        return False
    except Exception as e:
        print(f"Произошла ошибка во время конвертации: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Запуск SUMO симуляции для кругового движения и сбор данных.")
    parser.add_argument("--config-name", type=str, required=True, choices=["600m", "1k", "10k"],
                        help="Название конфигурации круга (например, 600m, 1k, 10k).")
    parser.add_argument("--max-num-vehicles", type=int, required=True,
                        help="Максимальное количество транспортных средств.")
    parser.add_argument("--tau", type=float, default=None,
                        help="Значение параметра tau (T headway) для модели IDM. Если не указано, используется значение из файла.")
    parser.add_argument("--output-dir", type=str, default="results/circle_simulation",
                        help="Директория для сохранения результатов симуляции.")
    parser.add_argument("--sumo-binary", type=str, default="sumo",
                        help="Путь к исполняемому файлу SUMO (например, sumo, sumo-gui, или полный путь).")
    parser.add_argument("--sumo-tools-dir", type=str, default="",
                        help="Путь к директории 'tools' в SUMO. По умолчанию используется $SUMO_HOME/tools, если SUMO_HOME установлен.")
    parser.add_argument("--simulation-duration", type=float, default=2000.0, help="Длительность симуляции в секундах SUMO.")
    # Добавляем аргумент для шага симуляции
    parser.add_argument("--step-length", type=float, default=0.1, help="Длина шага симуляции SUMO (в секундах), используется VSLController.")
    # --- ДОБАВЛЕНЫ АРГУМЕНТЫ ДЛЯ VSL ---
    parser.add_argument(
        "--vsl",
        action="store_true",
        help="Включить управление VSL (требует vsl_controller.py в корне проекта)."
    )
    parser.add_argument(
        "--vsl-params-json",
        type=str,
        help="JSON строка с параметрами для VSLController."
    )
    # --- КОНЕЦ ДОБАВЛЕННЫХ АРГУМЕНТОВ ---

    args = parser.parse_args()

    if not args.sumo_tools_dir or not os.path.isdir(args.sumo_tools_dir):
        print("Ошибка: Директория SUMO tools не найдена или не указана. ($SUMO_HOME может быть не установлен)")
        print("Пожалуйста, укажите корректный путь с помощью --sumo-tools-dir.")
        return

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) 
    base_config_abs_path = os.path.join(project_root, "config", "circles")

    original_sumocfg_filename = f"{args.config_name}.sumocfg"
    original_sumocfg_abs_path = os.path.join(base_config_abs_path, original_sumocfg_filename)
    original_routes_filename = "input_routes.rou.xml" # Имя файла маршрутов по умолчанию
    original_routes_abs_path = os.path.join(base_config_abs_path, original_routes_filename)

    if not os.path.exists(original_sumocfg_abs_path):
        print(f"Ошибка: Файл конфигурации {original_sumocfg_abs_path} не найден.")
        return
    if not os.path.exists(original_routes_abs_path):
        print(f"Ошибка: Файл маршрутов {original_routes_abs_path} не найден.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_output_dir_for_sim_files = os.path.abspath(os.path.join(args.output_dir, f"{args.config_name}_{args.max_num_vehicles}_vehicles_{timestamp}"))
    os.makedirs(current_run_output_dir_for_sim_files, exist_ok=True)

    # ----- Настройка детекторов -----
    num_detectors = int(np.floor(FIXED_RING_LENGTH / DETECTOR_SPACING_M))
    if num_detectors == 0 and FIXED_RING_LENGTH > 0: # Хотя бы 1 детектор, если кольцо есть
        num_detectors = 1
    elif num_detectors == 0:
        print("Длина кольца или расстояние между детекторами не позволяют разместить детекторы.")
        return
    
    actual_detector_spacing = FIXED_RING_LENGTH / num_detectors if num_detectors > 0 else 0
    detector_positions = [i * actual_detector_spacing for i in range(num_detectors)]
    # Округляем до 2 знаков после запятой, чтобы избежать проблем с SUMO из-за слишком длинных float
    detector_positions = [round(pos, 2) for pos in detector_positions]
    # Убедимся, что последняя позиция не превышает длину ребра (она должна быть < длины)
    if detector_positions and detector_positions[-1] >= FIXED_RING_LENGTH:
        detector_positions[-1] = FIXED_RING_LENGTH - 0.01 # Чуть меньше длины ребра
        
    detector_ids = [f"det_{i}" for i in range(num_detectors)]
    cameras_coords_np = np.array(detector_positions)

    # Генерируем additional файл с детекторами во временной/результирующей директории
    # Этот файл будет лежать рядом с временным .sumocfg
    detector_add_file_abs_path = generate_detector_additional_file(
        current_run_output_dir_for_sim_files, 
        RING_EDGE_ID, 
        num_detectors, 
        detector_positions, 
        detector_ids, 
        DETECTOR_SAMPLING_PERIOD_S
    )
    # Для sumocfg нужен basename, т.к. он будет в той же директории
    detector_add_file_basename = os.path.basename(detector_add_file_abs_path) 
    # --------------------------------

    temp_rou_filename = f"temp_routes_{timestamp}.rou.xml"
    temp_rou_file_abs_path = os.path.join(current_run_output_dir_for_sim_files, temp_rou_filename)
    modify_routes_file(original_routes_abs_path, temp_rou_file_abs_path, args.max_num_vehicles, args.tau)

    temp_sumocfg_filename = f"temp_config_{timestamp}.sumocfg"
    temp_sumocfg_file_abs_path = os.path.join(current_run_output_dir_for_sim_files, temp_sumocfg_filename)
    
    # Определяем длительность симуляции (например, из .sumocfg или задаем явно)
    # В modify_sumocfg_file мы устанавливаем end="2000", так что используем это.
    simulation_duration_s_from_cfg = 2000.0 

    modify_sumocfg_file(original_sumocfg_abs_path, temp_sumocfg_file_abs_path, 
                        temp_rou_filename, args.max_num_vehicles, 
                        os.path.join(current_run_output_dir_for_sim_files, f"fcd_output_{args.config_name}_{timestamp}.xml"),
                        base_config_abs_path,
                        detector_add_file_basename=detector_add_file_basename) # Передаем имя файла детекторов

    fcd_xml_output_filename_for_conversion = f"fcd_output_{args.config_name}_{timestamp}.xml"
    fcd_output_file_for_conversion = os.path.join(current_run_output_dir_for_sim_files, fcd_xml_output_filename_for_conversion)
    csv_output_filename_generated = f"fcd_output_{args.config_name}_{timestamp}.csv"
    csv_output_file_generated_abs_path = os.path.join(current_run_output_dir_for_sim_files, csv_output_filename_generated)

    # --- НАЧАЛО БЛОКА ИЗМЕНЕНИЙ ДЛЯ VSL В MAIN ---
    controller_init_params = None # Инициализируем как None
    if args.vsl:
        if not VSL_CONTROLLER_AVAILABLE:
            print("ОШИБКА VSL: VSLController не был импортирован. VSL не будет запущен.")
        elif not args.vsl_params_json:
            print("ОШИБКА VSL: Флаг --vsl указан, но --vsl-params-json не предоставлен. VSL не будет запущен.")
        else:
            try:
                vsl_params_from_json = json.loads(args.vsl_params_json)
                print(f"Попытка инициализации VSLController с параметрами из JSON: {vsl_params_from_json}")

                required_keys = [
                    "idm_v0_default", "ctrl_segments_lanes", "ts_control_interval",
                    "kp", "ki", "kd", "v_min_vsl_limit", "rho_crit_target",
                    "vsl_detector_id", "vsl_detector_length_m", "log_csv_filename"
                ]
                missing_keys = [key for key in required_keys if key not in vsl_params_from_json]

                if missing_keys:
                    print(f"ОШИБКА VSL: Отсутствуют обязательные параметры VSL в --vsl-params-json: {missing_keys}")
                else:
                    vsl_log_full_path = os.path.join(current_run_output_dir_for_sim_files, vsl_params_from_json["log_csv_filename"])
                    print(f"Путь для лога VSL: {vsl_log_full_path}")

                    controller_init_params = {
                        "traci_conn": None, # Будет установлено в run_simulation после запуска traci
                        "idm_v0_default": vsl_params_from_json["idm_v0_default"],
                        "ctrl_segments_lanes": vsl_params_from_json["ctrl_segments_lanes"],
                        "ts_control_interval": vsl_params_from_json["ts_control_interval"],
                        "kp": vsl_params_from_json["kp"],
                        "ki": vsl_params_from_json["ki"],
                        "kd": vsl_params_from_json["kd"],
                        "v_min_vsl_limit": vsl_params_from_json["v_min_vsl_limit"],
                        "rho_crit_target": vsl_params_from_json["rho_crit_target"],
                        "vsl_detector_id": vsl_params_from_json["vsl_detector_id"],
                        "vsl_detector_length_m": vsl_params_from_json["vsl_detector_length_m"],
                        "log_csv_full_path": vsl_log_full_path,
                        "sim_step_length": args.step_length, # <<< ИСПОЛЬЗУЕМ args.step_length
                        "enabled": True
                    }
                    print("Параметры для VSLController подготовлены.")
            except json.JSONDecodeError:
                print("ОШИБКА VSL: Не удалось декодировать JSON из --vsl-params-json.")
            except KeyError as e:
                print(f"ОШИБКА VSL: Отсутствует ключ {e} в параметрах VSL из JSON.")
            except Exception as e:
                print(f"ОШИБКА VSL: Непредвиденная ошибка при обработке параметров VSL: {e}")
    # --- КОНЕЦ БЛОКА ИЗМЕНЕНИЙ ДЛЯ VSL В MAIN ---

    # Запускаем симуляцию
    # Передаем controller_init_params как vsl_init_params в run_simulation
    simulation_successful, rt_wave_detector_instance = run_simulation(
        args.sumo_binary,
        temp_sumocfg_file_abs_path,
        current_run_output_dir_for_sim_files,
        args.config_name,
        timestamp,
        detector_ids,
        cameras_coords_np, # detector_positions переименован в cameras_coords_np
        DETECTOR_SAMPLING_PERIOD_S,
        simulation_duration_s_from_cfg, # Используем извлеченное из sumocfg время
        vsl_init_params=controller_init_params # Передаем параметры для инициализации VSL
    )

    if simulation_successful:
        print(f"Симуляция завершена, приступаем к конвертации FCD и сохранению данных детекторов.")
        
        if os.path.exists(fcd_output_file_for_conversion):
            print(f"Конвертация FCD XML ({fcd_output_file_for_conversion}) в CSV ({csv_output_file_generated_abs_path})...")
            convert_fcd_xml_to_csv(fcd_output_file_for_conversion, csv_output_file_generated_abs_path, args.sumo_tools_dir)
        else:
            print(f"FCD output file не найден ({fcd_output_file_for_conversion}), конвертация в CSV не будет выполнена.")

        # Сохранение данных детекторов RTWD и конфигурации, если RTWD был активен
        save_rtwd_data_and_config(
            rt_wave_detector=rt_wave_detector_instance, 
            output_dir=current_run_output_dir_for_sim_files, # Используем правильную директорию
            detector_ids=detector_ids, # Передаем оригинальные detector_ids
            cameras_coords=cameras_coords_np, # Передаем cameras_coords_np
            detector_sampling_period=DETECTOR_SAMPLING_PERIOD_S # Передаем DETECTOR_SAMPLING_PERIOD_S
        )
        
    else:
        print(f"Симуляция НЕ УДАЛАСЬ для {temp_sumocfg_file_abs_path}")
        # Здесь можно добавить дополнительную логику обработки неудачной симуляции, если нужно
    
    # Удаление временных файлов .add.xml, .rou.xml, .sumocfg
    # ... (код удаления временных файлов без изменений) ...

# --- НОВАЯ ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ ДАННЫХ RTWD И КОНФИГУРАЦИИ ---
def save_rtwd_data_and_config(rt_wave_detector, output_dir, detector_ids, cameras_coords, detector_sampling_period):
    """
    Сохраняет данные от RealTimeWaveDetector (если они есть) и связанную конфигурацию.
    """
    # Убедимся, что директория вывода существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория для вывода RTWD: {output_dir}")

    if rt_wave_detector and hasattr(rt_wave_detector, 'V_data') and rt_wave_detector.V_data is not None and len(rt_wave_detector.V_data) > 0:
        print(f"Сохранение данных rt_wave_detector (V_data, N_data и др.) в .npy файлы в {output_dir}...")
        np.save(os.path.join(output_dir, 'V_data_rtwd.npy'), rt_wave_detector.V_data)
        np.save(os.path.join(output_dir, 'N_data_rtwd.npy'), rt_wave_detector.N_data)
        np.save(os.path.join(output_dir, 'Q_data_rtwd.npy'), rt_wave_detector.Q_data)
        np.save(os.path.join(output_dir, 'Rho_data_rtwd.npy'), rt_wave_detector.Rho_data)
        np.save(os.path.join(output_dir, 'time_data_rtwd.npy'), rt_wave_detector.time_data)
        
        # Сохраняем конфигурацию детекторов, которая использовалась RTWD
        # detector_ids (список строк), cameras_coords (numpy array), detector_sampling_period (число)
        np.save(os.path.join(output_dir, 'detector_ids_rtwd.npy'), np.array(detector_ids, dtype=object)) 
        np.save(os.path.join(output_dir, 'cameras_coords_rtwd.npy'), cameras_coords) 
        with open(os.path.join(output_dir, 'Ts_rtwd.txt'), 'w') as f_ts:
            f_ts.write(str(detector_sampling_period))
        print(f"Данные и конфигурация RTWD сохранены в: {output_dir}")

    else:
        print(f"Предупреждение: Данные rt_wave_detector не были инициализированы или не содержат данных. Будут созданы пустые индикаторные файлы в {output_dir}.")
        # Создаем пустые файлы, чтобы последующие скрипты не падали, если ожидают их
        # Добавляем суффикс _rtwd, чтобы не конфликтовать с другими возможными файлами
        for data_name in ['V_data_rtwd', 'N_data_rtwd', 'Q_data_rtwd', 'Rho_data_rtwd', 'time_data_rtwd', 'detector_ids_rtwd', 'cameras_coords_rtwd']:
            target_npy_path = os.path.join(output_dir, f'{data_name}.npy')
            if not os.path.exists(target_npy_path): # Создаем только если еще не существует
                 np.save(target_npy_path, np.array([]))
        
        target_txt_path = os.path.join(output_dir, 'Ts_rtwd.txt')
        if not os.path.exists(target_txt_path):
            with open(target_txt_path, 'w') as f_ts:
                f_ts.write(str(detector_sampling_period)) # Записываем плановый Ts для справки
        print(f"Пустые индикаторные файлы RTWD созданы в: {output_dir}")

if __name__ == "__main__":
    main() 