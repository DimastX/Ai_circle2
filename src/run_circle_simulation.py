import os
import subprocess
import argparse
import xml.etree.ElementTree as ET
import tempfile
import shutil

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
                          original_config_dir_abs_path):
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
    if additional_files_element is not None:
        original_add_file_name = additional_files_element.get("value")
        additional_files_element.set("value", os.path.join(original_config_dir_abs_path, original_add_file_name))
    else:
        # Это может быть необязательным элементом
        print("Предупреждение: элемент <additional-files> не найден в .sumocfg. Пропускаем.")

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
    fcd_output_element.set("value", fcd_output_file_abs_path)
    
    fcd_geo_element = output_element.find(".//fcd-output.geo")
    if fcd_geo_element is None:
        fcd_geo_element = ET.SubElement(output_element, "fcd-output.geo")
    fcd_geo_element.set("value", "true")

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


def run_simulation(sumo_binary, temp_sumocfg_file, results_dir, config_name, timestamp):
    """Запускает симуляцию SUMO."""
    # simulation_log_file = os.path.join(results_dir, f"sumo_log_{config_name}_{timestamp}.txt") # Не используется активно
    cmd = [sumo_binary, "-c", temp_sumocfg_file, "--no-warnings", "true"]
    print(f"Запуск SUMO: {' '.join(cmd)}")
    try:
        # Увеличиваем время ожидания, если симуляции долгие (хотя для 10 машин это не должно быть проблемой)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(temp_sumocfg_file))
        stdout, stderr = process.communicate(timeout=300) # Таймаут 5 минут

        if process.returncode != 0:
            print(f"Ошибка при запуске SUMO (код возврата: {process.returncode}):")
            if stdout:
                print("STDOUT:")
                print(stdout.decode(errors='replace'))
            if stderr:
                print("STDERR:")
                print(stderr.decode(errors='replace'))
            return False
        else:
            print("Симуляция SUMO успешно завершена.")
            if stdout:
                 print("STDOUT (SUMO):")
                 print(stdout.decode(errors='replace'))
            if stderr: 
                print("STDERR (SUMO - может содержать предупреждения или инфо):")
                print(stderr.decode(errors='replace'))
            return True
    except subprocess.TimeoutExpired:
        print("Ошибка: Симуляция SUMO превысила лимит времени.")
        process.kill()
        stdout, stderr = process.communicate()
        if stdout: print(f"STDOUT (timeout):\n{stdout.decode(errors='replace')}")
        if stderr: print(f"STDERR (timeout):\n{stderr.decode(errors='replace')}")
        return False
    except FileNotFoundError:
        print(f"Ошибка: {sumo_binary} не найден. Укажите правильный путь с помощью --sumo-binary.")
        return False
    except Exception as e:
        print(f"Произошла ошибка во время симуляции: {e}")
        return False

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

    args = parser.parse_args()

    if not args.sumo_tools_dir or not os.path.isdir(args.sumo_tools_dir):
        print("Ошибка: Директория SUMO tools не найдена или не указана. ($SUMO_HOME может быть не установлен)")
        print("Пожалуйста, укажите корректный путь с помощью --sumo-tools-dir.")
        return

    # Определяем абсолютный путь к директории с оригинальными конфигами (config/circles)
    # Это нужно для корректного разрешения путей к .net.xml и .add.xml
    # Предполагается, что скрипт запускается из корневой директории проекта, где есть 'config/circles'
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) # Предполагаем, что src находится в корне проекта
    base_config_abs_path = os.path.join(project_root, "config", "circles")
    # Если это не так, то более надежно будет, если пользователь запускает скрипт из корня проекта
    # и мы используем относительные пути оттуда.
    # Для простоты, будем считать, что getcwd() при запуске - это корень проекта.
    if not os.path.isdir(os.path.join(os.getcwd(), "config", "circles")):
        print("Ошибка: не удается найти директорию config/circles относительно текущей рабочей директории.")
        print("Пожалуйста, запускайте скрипт из корневой директории вашего проекта.")
        base_config_abs_path = os.path.abspath("config/circles") # Попытка, если структура другая
        if not os.path.isdir(base_config_abs_path):
             print(f"Директория {base_config_abs_path} также не найдена. Проверьте структуру проекта.")
             return
    else:
        base_config_abs_path = os.path.abspath(os.path.join(os.getcwd(), "config", "circles"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_with_timestamp_abs = os.path.abspath(os.path.join(args.output_dir, f"{args.config_name}_{args.max_num_vehicles}_vehicles_{timestamp}"))
    os.makedirs(results_dir_with_timestamp_abs, exist_ok=True)
    
    original_rou_file_abs = os.path.join(base_config_abs_path, "input_routes.rou.xml")
    original_sumocfg_file_abs = os.path.join(base_config_abs_path, f"{args.config_name}.sumocfg")

    if not os.path.isfile(original_rou_file_abs):
        print(f"Ошибка: Файл маршрутов {original_rou_file_abs} не найден.")
        return
    if not os.path.isfile(original_sumocfg_file_abs):
        print(f"Ошибка: Файл конфигурации {original_sumocfg_file_abs} не найден.")
        return

    temp_dir = tempfile.mkdtemp()
    try:
        temp_rou_filename = "temp_routes.rou.xml"
        temp_rou_file_abs = os.path.join(temp_dir, temp_rou_filename)
        temp_sumocfg_filename = "temp_config.sumocfg"
        temp_sumocfg_file_abs = os.path.join(temp_dir, temp_sumocfg_filename)
        
        # Имя FCD XML файла (должно быть внутри results_dir_with_timestamp_abs)
        # Обновлено для соответствия ожиданиям eidm_stability_analysis.py
        fcd_xml_output_filename = f"fcd_output_{args.config_name}_{args.max_num_vehicles}_{timestamp}.xml" 
        fcd_xml_output_file_abs_path = os.path.join(results_dir_with_timestamp_abs, fcd_xml_output_filename)

        # Передаем fcd_xml_output_file_abs_path в modify_sumocfg_file
        # Передаем args.tau в modify_routes_file
        modify_routes_file(original_rou_file_abs, temp_rou_file_abs, args.max_num_vehicles, args.tau)
        modify_sumocfg_file(original_sumocfg_file_abs, temp_sumocfg_file_abs,
                              temp_rou_filename, args.max_num_vehicles,
                              fcd_xml_output_file_abs_path, # Это путь к XML файлу, который SUMO должен создать
                              base_config_abs_path)

        # Имя для CSV файла (будет в той же директории, что и XML)
        # Обновлено для соответствия ожиданиям eidm_stability_analysis.py
        csv_output_filename = f"fcd_output_{args.config_name}_{args.max_num_vehicles}_{timestamp}.csv"
        csv_output_file_abs_path = os.path.join(results_dir_with_timestamp_abs, csv_output_filename)

        if run_simulation(args.sumo_binary, temp_sumocfg_file_abs, results_dir_with_timestamp_abs, args.config_name, timestamp):
            if convert_fcd_xml_to_csv(fcd_xml_output_file_abs_path, csv_output_file_abs_path, args.sumo_tools_dir):
                print(f"Симуляция и конвертация завершены. Результаты в: {results_dir_with_timestamp_abs}")
                print(f"CSV файл сохранен как: {csv_output_file_abs_path}")
            else:
                print(f"Симуляция завершена, но конвертация в CSV не удалась. XML файл: {fcd_xml_output_file_abs_path}")
        else:
            print("Ошибка во время выполнения симуляции SUMO.")

    finally:
        print(f"Очистка временной директории: {temp_dir}")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    from datetime import datetime 
    main() 