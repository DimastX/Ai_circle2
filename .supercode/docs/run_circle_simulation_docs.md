# Документация по `src/run_circle_simulation.py`

## Общее описание

Этот скрипт предназначен для запуска одиночных симуляций SUMO на кольцевой трассе. Он управляет генерацией конфигурационных файлов, запуском симулятора, сбором данных (FCD и с детекторов E1/E2) и, опционально, работой VSL-контроллера и детектора волн в реальном времени.

Основные задачи:

1.  **Генерация конфигурационных файлов SUMO**:
    *   Создает временные файлы маршрутов (`.rou.xml`) на основе шаблона и заданного количества ТС.
    *   Создает файл дополнительных элементов (`.add.xml`) для описания детекторов (E1 loop detectors).
    *   Модифицирует основной конфигурационный файл SUMO (`.sumocfg`), указывая пути к сгенерированным файлам маршрутов, детекторов и пути для вывода FCD данных.
2.  **Запуск симуляции SUMO**:
    *   Запускает SUMO (GUI или CLI) с использованием сгенерированной конфигурации.
    *   Управляет симуляционным временем.
3.  **Взаимодействие через TraCI** (с использованием `src/traci_interaction.py`):
    *   Подключается к SUMO через TraCI.
    *   На каждом шаге симуляции:
        *   Считывает данные с детекторов E1 (скорость, количество ТС, плотность, поток).
        *   (Опционально) Обновляет `RealTimeWaveDetector` из `traci_interaction.py` данными с детекторов и проверяет на наличие stop-and-go волн.
        *   (Опционально) Вызывает шаг `VSLController` из `src/vsl_controller.py`, если VSL активирован.
4.  **Сбор и сохранение данных**:
    *   Агрегирует данные с детекторов E1 и сохраняет их в CSV файл (`detector_data.csv`).
    *   SUMO сохраняет FCD данные (траектории всех ТС) в XML файл.
    *   (Опционально) Конвертирует FCD XML в CSV формат (`fcd_output.csv`) с помощью `xml2csv.py` из `SUMO_HOME/tools`.
    *   (Опционально) `RealTimeWaveDetector` сохраняет обнаруженные события волн в `rt_detected_wave_events.csv`.
    *   (Опционально) `VSLController` сохраняет лог своей работы в `vsl_controller_log.csv`.
5.  **Управление VSL** (опционально):
    *   Если передан флаг `--vsl` и JSON-строка с параметрами `--vsl-params-json`, инициализирует и запускает `VSLController`.

## Ключевые функции и их назначение

-   **`generate_detector_additional_file`**: Создает `.add.xml` с определениями детекторов E1.
-   **`modify_routes_file`**: Создает временный файл маршрутов (`.rou.xml`) с заданным числом ТС и параметрами отправления.
-   **`modify_sumocfg_file`**: Создает временный `.sumocfg` файл, связывая все необходимые файлы и пути вывода.
-   **`run_simulation`**: Основная функция, управляющая запуском симуляции, взаимодействием через TraCI и сбором данных. Внутри нее происходит инициализация и работа `RealTimeWaveDetector` и `VSLController`.
-   **`convert_fcd_xml_to_csv`**: Вызывает `xml2csv.py` для конвертации FCD данных.
-   **`save_rtwd_data_and_config`**: Сохраняет данные и конфигурацию `RealTimeWaveDetector`.
-   **`main`**: Парсит аргументы командной строки, настраивает пути, вызывает генерацию конфигураций и `run_simulation`.

## Внешние зависимости (ключевые импорты)

-   **Стандартные библиотеки Python**:
    *   `os`
    *   `subprocess`
    *   `argparse`
    *   `datetime`
    *   `xml.etree.ElementTree`
    *   `csv`
    *   `json`
    *   `shutil`
    *   `time`
    *   `logging`
-   **Сторонние библиотеки (из `requirements.txt`)**:
    *   `numpy` (для `RealTimeWaveDetector` и `VSLController`)
    *   `pandas` (для `RealTimeWaveDetector` и `VSLController`)
-   **SUMO (TraCI)**:
    *   `traci` (импортируется, если доступен)
-   **Внутренние модули проекта**:
    *   `src.traci_interaction` (импортирует `connect_sumo`, `simulation_step`, `get_detector_data`, `close_sumo`, `RealTimeWaveDetector`)
    *   `src.vsl_controller` (импортирует `VSLController`)

## Взаимодействие с другими компонентами

-   Вызывается скриптом `src/eidm_stability_analysis.py` для проведения отдельных симуляционных экспериментов.
-   Использует `src/traci_interaction.py` для всего низкоуровневого взаимодействия с SUMO через TraCI и для детектора волн.
-   Использует `src/vsl_controller.py` для реализации логики VSL, если та активирована.
-   Результаты работы этого скрипта (CSV файлы с данными) анализируются `src/analyze_circle_data.py`.

## Основные сценарии вызова (из `eidm_stability_analysis.py`)

Скрипт обычно не запускается вручную, а вызывается из `eidm_stability_analysis.py`.

1.  **Симуляция без VSL**:
    `eidm_stability_analysis.py` вызывает `run_circle_simulation.py` с параметрами, определяющими количество ТС, пути к SUMO и директорию вывода. Флаг `--vsl` не передается.

2.  **Симуляция с VSL**:
    `eidm_stability_analysis.py` вызывает `run_circle_simulation.py` аналогично, но с добавлением флага `--vsl` и параметра `--vsl-params-json`, содержащего конфигурацию VSL-контроллера.

## Ключевые структуры данных и файлы

-   **Входные файлы (шаблоны)**:
    *   `config/circles/{config_name}/{config_name}.net.xml` (сеть)
    *   `config/circles/{config_name}/{config_name}.rou.xml` (шаблон маршрутов)
    *   `config/circles/{config_name}/{config_name}.sumocfg` (шаблон конфигурации SUMO)
-   **Генерируемые временные файлы**:
    *   `temp_{config_name}_{num_vehicles}.rou.xml`
    *   `temp_{config_name}_detectors.add.xml`
    *   `temp_{config_name}_{num_vehicles}.sumocfg`
-   **Выходные файлы (в `output_dir` / `results_dir_with_timestamp_abs`)**:
    *   `fcd_output_{...}.xml` (или `.csv` после конвертации): траектории ТС.
    *   `detector_data.csv`: агрегированные данные с детекторов E1.
    *   `rt_config.json` (от `RealTimeWaveDetector`)
    *   `rt_detected_wave_events.csv` (от `RealTimeWaveDetector`)
    *   `vsl_controller_log.csv` (от `VSLController`, если активен)
    *   `simulation_params.json`: сводка по параметрам запуска симуляции.

## Конфигурация и параметры командной строки

-   `--config-name`: Имя конфигурации кольцевой трассы (например, `1k`).
-   `--max-num-vehicles`: Максимальное (или точное, в зависимости от логики) количество ТС.
-   `--tau-value`: Параметр $\tau$ для распределения Вейбулла (время между ТС).
-   `--output-dir`: Основная директория для сохранения результатов этого запуска.
-   `--sumo-binary`: Путь к `sumo-gui.exe` или `sumo.exe`.
-   `--sumo-tools-dir`: Путь к директории `tools` в SUMO.
-   `--simulation-duration`: Длительность симуляции в секундах.
-   `--detector-sampling-period`: Период опроса детекторов E1.
-   `--enable-realtime-wave-detection`: Флаг для активации `RealTimeWaveDetector`.
-   `--vsl`: Флаг для активации VSL.
-   `--vsl-params-json`: JSON-строка с параметрами для `VSLController`.

## Замечания и возможные улучшения

-   Логика определения `num_vehicles` на основе `max-num-vehicles` и `tau-value` может потребовать уточнений для различных сценариев (фиксированное число ТС vs. генерация по потоку).
-   Передача параметров IDM (если они меняются для каждой симуляции, как в `eidm_stability_analysis.py`) в настоящее время не реализована через аргументы командной строки `run_circle_simulation.py`. Он использует параметры IDM, зашитые в файлы `.rou.xml` или глобальные настройки SUMO. Это может быть точкой для улучшения, чтобы сделать `run_circle_simulation.py` более гибким к изменениям параметров модели поведения ТС. 