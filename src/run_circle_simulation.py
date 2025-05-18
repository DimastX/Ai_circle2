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

# Импорт из нашего нового модуля для взаимодействия с TraCI
# Предполагаем, что traci_interaction.py находится в том же каталоге src
import traci_interaction as ti

# Константы для детекторов
RING_EDGE_ID = "a"  # ID кольцевого ребра из 1k.net.xml
FIXED_RING_LENGTH = 901.53  # Длина кольца, как в eidm_stability_analysis.py
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
                   detector_ids, cameras_coords, detector_sampling_period, simulation_duration_s):
    """Запускает симуляцию SUMO с использованием TraCI и собирает данные с детекторов."""
    
    sumo_cmd = [sumo_binary, "-c", temp_sumocfg_file, "--no-warnings", "true"]
    # Для TraCI лучше запускать SUMO с определенным портом, чтобы избежать конфликтов,
    # но traci.start() может выбрать его автоматически.
    # Если используем traci.start(), то remote-port не нужен в sumo_cmd.
    # Если используем traci.init(port), то нужен и порт должен быть свободен.
    # traci_interaction.connect_sumo(sumo_cmd) будет использовать traci.start()

    print(f"Подготовка к запуску SUMO с TraCI: {' '.join(sumo_cmd)}")

    num_detector_steps = int(simulation_duration_s / detector_sampling_period)
    num_cameras = len(detector_ids)
    
    # Инициализация массивов для хранения данных
    # V_data: (num_cameras, num_detector_steps) - средние скорости
    # N_data: (num_cameras, num_detector_steps) - количество машин
    V_data_collected = np.full((num_cameras, num_detector_steps), np.nan)
    N_data_collected = np.full((num_cameras, num_detector_steps), np.nan)
    Q_data_collected = np.full((num_cameras, num_detector_steps), np.nan) # Для интенсивности
    rho_data_collected = np.full((num_cameras, num_detector_steps), np.nan) # Для плотности
    
    actual_simulation_time_points = [] # Для отладки или если нужно точное время каждого сбора

    # +++ Инициализация RealTimeWaveDetector +++
    wave_detector = ti.RealTimeWaveDetector(
        detector_ids_ordered=detector_ids,
        cameras_coords=cameras_coords, # Передаем координаты
        Ts=detector_sampling_period
        # Остальные параметры (buffer_size_seconds и т.д.) будут использованы по умолчанию из traci_interaction.py
    )
    # ++++++++++++++++++++++++++++++++++++++++++

    if not ti.connect_sumo(sumo_cmd):
        print("Не удалось запустить SUMO или подключиться через TraCI.")
        return False, None, None, None, None, None # Добавляем None для Q и rho

    simulation_successful = True
    try:
        current_detector_step_idx = 0
        for step in range(int(simulation_duration_s * 1000)): # Симулируем с шагом 1 мс (по умолчанию в TraCI)
            ti.simulation_step() # Выполняем один шаг симуляции TraCI (обычно 1 секунда модельного времени, если не настроено иначе)
            
            current_sim_time_s = traci.simulation.getTime()

            # Проверяем, не пора ли собирать данные с детекторов
            # Мы хотим собрать данные num_detector_steps раз.
            # Сбор происходит в конце каждого detector_sampling_period.
            if current_detector_step_idx < num_detector_steps and current_sim_time_s >= (current_detector_step_idx + 1) * detector_sampling_period - 1e-3: # Небольшой допуск для float сравнения
                
                detector_data_at_step = ti.get_detector_data(detector_ids)
                actual_simulation_time_points.append(current_sim_time_s)

                # +++ Вызов детектора волн в реальном времени +++
                if wave_detector:
                    wave_detector.update_and_detect(current_sim_time_s, detector_data_at_step)
                # ++++++++++++++++++++++++++++++++++++++++++++++

                for cam_idx, det_id in enumerate(detector_ids):
                    if det_id in detector_data_at_step:
                        current_n = detector_data_at_step[det_id]['vehicle_count']
                        current_v = detector_data_at_step[det_id]['mean_speed']
                        
                        N_data_collected[cam_idx, current_detector_step_idx] = current_n
                        V_data_collected[cam_idx, current_detector_step_idx] = current_v

                        # Расчет Q и rho
                        current_q = 0.0
                        if detector_sampling_period > 1e-9: # Избегаем деления на ноль
                            current_q = float(current_n) / detector_sampling_period
                        
                        current_rho = 0.0
                        if current_v > 1e-3: # Избегаем деления на ноль или очень малую скорость
                            current_rho = current_q / current_v
                        elif current_q > 1e-3: # Если есть поток, но скорость почти ноль
                            current_rho = np.nan # Плотность не определена или очень велика
                        
                        Q_data_collected[cam_idx, current_detector_step_idx] = current_q
                        rho_data_collected[cam_idx, current_detector_step_idx] = current_rho
                    else:
                        # Этого не должно произойти, если get_detector_data обрабатывает ошибки
                        N_data_collected[cam_idx, current_detector_step_idx] = 0 
                        V_data_collected[cam_idx, current_detector_step_idx] = np.nan # или 0, но nan лучше для V
                        Q_data_collected[cam_idx, current_detector_step_idx] = 0
                        rho_data_collected[cam_idx, current_detector_step_idx] = 0
                
                if current_detector_step_idx % 1000 == 0 or current_detector_step_idx == num_detector_steps -1: # Логируем каждые 20 шагов или последний
                    print(f"TraCI: Собраны данные с детекторов на шаге {current_detector_step_idx+1}/{num_detector_steps} (Время SUMO: {current_sim_time_s:.2f}s)")
                
                current_detector_step_idx += 1
            
            if current_sim_time_s >= simulation_duration_s - 1e-3:
                print(f"TraCI: Достигнута целевая длительность симуляции ({simulation_duration_s}s). Время SUMO: {current_sim_time_s:.2f}s")
                break
        
        if current_detector_step_idx < num_detector_steps:
            print(f"Предупреждение: Симуляция завершилась до сбора всех ({num_detector_steps}) запланированных данных с детекторов. Собрано: {current_detector_step_idx}")
            # Обрезаем массивы, если собрали меньше данных
            V_data_collected = V_data_collected[:, :current_detector_step_idx]
            N_data_collected = N_data_collected[:, :current_detector_step_idx]
            Q_data_collected = Q_data_collected[:, :current_detector_step_idx]
            rho_data_collected = rho_data_collected[:, :current_detector_step_idx]

    except traci.TraCIException as e:
        print(f"Произошла ошибка TraCI во время симуляции: {e}")
        simulation_successful = False
    except Exception as e:
        print(f"Произошла непредвиденная ошибка во время цикла симуляции TraCI: {e}")
        import traceback
        traceback.print_exc()
        simulation_successful = False
    finally:
        # +++ Сохранение данных о волнах из RealTimeWaveDetector перед закрытием TraCI +++
        if wave_detector and simulation_successful: # Только если детектор был и симуляция успешна (или частично успешна)
            rt_event_details = wave_detector.get_detected_event_details()
            if rt_event_details:
                rt_events_df = pd.DataFrame(rt_event_details)
                rt_csv_path = os.path.join(results_dir, "rt_detected_wave_events.csv")
                try:
                    rt_events_df.to_csv(rt_csv_path, index=False, encoding='utf-8')
                    print(f"Данные о волнах, обнаруженных RealTimeWaveDetector, сохранены в: {rt_csv_path}")
                except Exception as e_csv:
                    print(f"Ошибка при сохранении rt_detected_wave_events.csv: {e_csv}")
            else:
                print("RealTimeWaveDetector не обнаружил событий для сохранения в CSV.")
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ti.close_sumo()

    if simulation_successful:
        print("Симуляция SUMO с TraCI успешно завершена.")
        # Сохраняем данные детекторов
        np.save(os.path.join(results_dir, "V_data_detectors.npy"), V_data_collected)
        np.save(os.path.join(results_dir, "N_data_detectors.npy"), N_data_collected)
        np.save(os.path.join(results_dir, "Q_data_detectors.npy"), Q_data_collected) # Сохраняем Q
        np.save(os.path.join(results_dir, "rho_data_detectors.npy"), rho_data_collected) # Сохраняем rho
        
        # cameras_coords должны быть определены до вызова run_simulation и переданы, или вычислены в main
        # Для сохранения, они должны быть известны. Предположим, они передаются или доступны глобально для main
        # В main() они вычисляются как detector_positions.
        # Если run_simulation не знает о них, то сохранение coords и Ts здесь не совсем верно.
        # Перенесем сохранение cameras_coords и Ts_detector в main, после вызова run_simulation.

        print(f"Данные детекторов (V, N, Q, rho) сохранены в директории: {results_dir}")
        return True, V_data_collected, N_data_collected, Q_data_collected, rho_data_collected, actual_simulation_time_points
    else:
        print("Симуляция SUMO с TraCI завершилась с ошибками или не полностью.")
        return False, None, None, None, None, None

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
                        help="Путь к исполняемому файлу SUMO (sumo или sumo-gui).")
    parser.add_argument("--sumo-tools-dir", type=str, 
                        default=os.path.join(os.getenv("SUMO_HOME", ""), "tools") if os.getenv("SUMO_HOME") else "",
                        help="Путь к директории 'tools' в установке SUMO. По умолчанию пытается использовать $SUMO_HOME/tools.")
    parser.add_argument("--simulation-duration", type=int, default=1000, help="Длительность симуляции в секундах.")

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
    results_dir_with_timestamp_abs = os.path.abspath(os.path.join(args.output_dir, f"{args.config_name}_{args.max_num_vehicles}_vehicles_{timestamp}"))
    os.makedirs(results_dir_with_timestamp_abs, exist_ok=True)

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
        results_dir_with_timestamp_abs, 
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
    temp_rou_file_abs_path = os.path.join(results_dir_with_timestamp_abs, temp_rou_filename)
    modify_routes_file(original_routes_abs_path, temp_rou_file_abs_path, args.max_num_vehicles, args.tau)

    temp_sumocfg_filename = f"temp_config_{timestamp}.sumocfg"
    temp_sumocfg_file_abs_path = os.path.join(results_dir_with_timestamp_abs, temp_sumocfg_filename)
    
    # Определяем длительность симуляции (например, из .sumocfg или задаем явно)
    # В modify_sumocfg_file мы устанавливаем end="2000", так что используем это.
    simulation_duration_s_from_cfg = 2000.0 

    modify_sumocfg_file(original_sumocfg_abs_path, temp_sumocfg_file_abs_path, 
                        temp_rou_filename, args.max_num_vehicles, 
                        os.path.join(results_dir_with_timestamp_abs, f"fcd_output_{args.config_name}_{timestamp}.xml"),
                        base_config_abs_path,
                        detector_add_file_basename=detector_add_file_basename) # Передаем имя файла детекторов

    fcd_xml_output_filename = f"fcd_output_{args.config_name}_{timestamp}.xml"
    fcd_xml_output_file_abs_path = os.path.join(results_dir_with_timestamp_abs, fcd_xml_output_filename)
    csv_output_filename = f"fcd_output_{args.config_name}_{timestamp}.csv"
    csv_output_file_abs_path = os.path.join(results_dir_with_timestamp_abs, csv_output_filename)

    # Запуск симуляции и сбор данных с детекторов
    sim_ok, V_data, N_data, Q_data, rho_data, time_points = run_simulation(
        args.sumo_binary, 
        temp_sumocfg_file_abs_path, 
        results_dir_with_timestamp_abs, 
        args.config_name, 
        timestamp,
        detector_ids,                  # Список ID детекторов
        cameras_coords_np,             # Передаем numpy массив координат камер
        DETECTOR_SAMPLING_PERIOD_S,    # Период опроса
        simulation_duration_s_from_cfg # Общая длительность симуляции
    )

    if sim_ok:
        print("Симуляция завершена, приступаем к конвертации FCD и сохранению данных детекторов.")
        # Сохраняем данные детекторов, если они были собраны
        if V_data is not None and N_data is not None:
            v_data_path = os.path.join(results_dir_with_timestamp_abs, "V_data_detectors.npy")
            n_data_path = os.path.join(results_dir_with_timestamp_abs, "N_data_detectors.npy")
            q_data_path = os.path.join(results_dir_with_timestamp_abs, "Q_data_detectors.npy")
            rho_data_path = os.path.join(results_dir_with_timestamp_abs, "rho_data_detectors.npy")
            cameras_coords_path = os.path.join(results_dir_with_timestamp_abs, "cameras_coords.npy")
            ts_detector_path = os.path.join(results_dir_with_timestamp_abs, "Ts_detector.txt")

            try:
                np.save(v_data_path, V_data)
                np.save(n_data_path, N_data)
                np.save(q_data_path, Q_data)
                np.save(rho_data_path, rho_data)
                np.save(cameras_coords_path, cameras_coords_np) # Сохраняем массив numpy
                with open(ts_detector_path, 'w') as f_ts:
                    f_ts.write(str(DETECTOR_SAMPLING_PERIOD_S))
                print(f"Данные детекторов сохранены в: {results_dir_with_timestamp_abs}")
            except Exception as e:
                print(f"Ошибка при сохранении данных детекторов: {e}")
        else:
            print("Данные детекторов (V_data, N_data) не были получены от run_simulation.")

        # Конвертация FCD XML в CSV (если FCD вывод был включен и файл существует)
        if os.path.exists(fcd_xml_output_file_abs_path):
            if convert_fcd_xml_to_csv(fcd_xml_output_file_abs_path, csv_output_file_abs_path, args.sumo_tools_dir):
                print(f"Симуляция и конвертация FCD завершены. Результаты в: {results_dir_with_timestamp_abs}")
            else:
                print(f"Симуляция завершена, но произошла ошибка при конвертации FCD XML в CSV.")
        else:
            print(f"FCD XML файл ({fcd_xml_output_file_abs_path}) не найден. Пропуск конвертации.")

        # Сохраняем cameras_coords и Ts_detector здесь, так как они известны в main
        np.save(os.path.join(results_dir_with_timestamp_abs, "cameras_coords.npy"), cameras_coords_np)
        with open(os.path.join(results_dir_with_timestamp_abs, "Ts_detector.txt"), 'w') as f_ts:
            f_ts.write(str(DETECTOR_SAMPLING_PERIOD_S))
        print(f"Файлы cameras_coords.npy и Ts_detector.txt сохранены в {results_dir_with_timestamp_abs}")
    else:
        print(f"Симуляция не удалась или была прервана. Результаты могут быть неполными в: {results_dir_with_timestamp_abs}")

    # Очистка временных файлов (опционально, но может быть полезно)
    # try:
    #     if os.path.exists(temp_rou_file_abs_path): os.remove(temp_rou_file_abs_path)
    #     if os.path.exists(temp_sumocfg_file_abs_path): os.remove(temp_sumocfg_file_abs_path)
    #     # detector_add_file_abs_path создается в results_dir, так что его можно не удалять как временный
    # except OSError as e:
    #     print(f"Ошибка при удалении временных файлов: {e}")

if __name__ == "__main__":
    main() 