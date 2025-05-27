# AI Circle Simulation

Проект для моделирования движения транспортных средств по круговой дороге с использованием SUMO (Simulation of Urban MObility) и EIDM (Enhanced Intelligent Driver Model). Проект включает в себя различные сценарии моделирования, включая анализ пробок, теорию пробок и управление переменными скоростными ограничениями (VSL).

## Структура проекта

```
.
├── config/
│   ├── network/     # Файлы сети (circle.net.xml, circle.nod.xml, circle.edg.xml)
│   └── routes/      # Файлы маршрутов (генерируются автоматически)
│   └── vsl_config.yml # Конфигурационный файл для VSL контроллера
├── data/            # Может использоваться для хранения специфичных данных сценариев (не результатов)
├── results/         # Директория для всех результатов симуляций и анализа
│   └── run_XXXX/    # Конкретная директория запуска симуляции
│       ├── analysis_plots_YYYY/ # Графики и JSON из analyze_circle_data.py
│       ├── fcd_output_...csv    # FCD данные от SUMO
│       ├── V_data_detectors.npy # Данные о скорости с детекторов
│       ├── N_data_detectors.npy # Данные о количестве ТС с детекторов
│       ├── Q_data_detectors.npy # Данные о потоке с детекторов
│       ├── rho_data_detectors.npy# Данные о плотности с детекторов
│       ├── cameras_coords.npy   # Координаты детекторов
│       ├── Ts_detector.txt      # Период опроса детекторов
│       ├── rt_detected_wave_events.csv # События волн, обнаруженные в реальном времени
│       └── vsl_log.csv          # Лог работы VSL контроллера (если включено)
├── ring_files/      # Файлы для кольцевой дороги (например, add_detectors.xml)
├── src/
│   ├── __init__.py
│   ├── run_circle_simulation.py   # Главный скрипт запуска симуляций с детекторами
│   ├── analyze_circle_data.py    # Скрипт анализа данных FCD и детекторов
│   ├── eidm_stability_analysis.py# Теоретический анализ устойчивости и вызов симуляций
│   ├── traci_interaction.py      # Модуль для взаимодействия с TraCI и RealTimeWaveDetector
│   ├── vsl_controller.py         # Модуль VSL контроллера
│   ├── generate_circle_rou_new.py  # Генератор маршрутов (может быть устаревшим)
│   ├── run_circle.py              # Старый скрипт запуска симуляции (без детекторов)
│   ├── analyze_data.py           # Старый скрипт анализа данных
│   ├── new_breakdown.py          # Моделирование пробок (базовая версия, может быть устаревшим)
│   ├── new_breakdown_N.py        # Моделирование пробок (расширенная версия, может быть устаревшим)
│   ├── theory_breakdown.py       # Теоретический анализ пробок (может быть устаревшим)
│   └── circle_gen.py             # Генератор кольцевой дороги (может быть устаревшим)
└── requirements.txt # Зависимости проекта
```
(Некоторые старые скрипты в `src/` могут быть неактуальны или заменены новой логикой в `run_circle_simulation.py`, `analyze_circle_data.py` и `eidm_stability_analysis.py`)

## Установка

1. Установите SUMO (Simulation of Urban MObility):
   - Windows: Скачайте и установите с [официального сайта](https://sumo.dlr.de/docs/Installing/Windows.html)
   - Linux: `sudo apt-get install sumo`

2. Установите зависимости проекта:
   ```bash
   pip install -r requirements.txt
   ```

## Использование

Основной рабочий процесс теперь сосредоточен вокруг `src/eidm_stability_analysis.py` для теоретического анализа и запуска серий симуляций, или `src/run_circle_simulation.py` для запуска отдельных симуляций с детекторами.

### Запуск одиночной симуляции с детекторами и анализом
1. Запуск симуляции (пример):
   ```bash
   cd src
   python run_circle_simulation.py --num-vehicles 77 --sumo-binary path/to/sumo-gui.exe --simulation-duration 1800
   ```
   (Или `sumo.exe` для CLI)
   Это создаст директорию в `results/` с именем, включающим параметры и временную метку. Внутри будут FCD CSV, данные детекторов (`.npy`, `.txt`) и `rt_detected_wave_events.csv`.

2. Анализ результатов:
   ```bash
   cd src
   python analyze_circle_data.py --results-dir path/to/results/run_XXXX --warmup-time 150
   ```
   Это создаст поддиректорию `analysis_plots_YYYY` внутри `run_XXXX` с графиками и `analysis_summary.json`.

### Запуск симуляции с VSL контроллером

VSL контроллер интегрируется в основной симуляционный скрипт (`run_circle_simulation.py`). Это позволяет активировать VSL при запуске симуляций, в том числе через `src/eidm_stability_analysis.py` с флагом `--vsl --run-sumo-simulations`.

Для интеграции в симуляционный скрипт, имеющий главный цикл, `VSLController` импортируется и используется следующим образом:

    ```python
    # В вашем основном скрипте симуляции
    import traci
    from src.vsl_controller import VSLController, Config  # Убедитесь, что src в PYTHONPATH

    # ... ваш код инициализации TraCI ...

    # Пример получения флага --vsl из аргументов
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--vsl', action='store_true', help='Enable VSL controller')
    # args = parser.parse_args()
    # vsl_enabled_by_flag = args.vsl

    vsl_enabled_by_flag = True # или False, в зависимости от вашей логики

    if vsl_enabled_by_flag:
        vsl_config_path = "vsl_config.yml" # Путь к вашему конфиг файлу
        vsl_cfg = Config(vsl_config_path)
        vsl_controller = VSLController(traci, vsl_cfg, enabled=True)
    else:
        vsl_controller = None

    simulation_time = 0
    step_length = 0.1 # Ваш шаг симуляции

    while simulation_time < SIMULATION_DURATION:
        traci.simulationStep()
        simulation_time += step_length

        if vsl_controller and vsl_controller.is_enabled():
            vsl_controller.step(simulation_time)

        # ... остальная логика вашего шага симуляции ...

    # ... ваш код завершения TraCI ...
    ```

### Теоретический анализ устойчивости и пакетный запуск симуляций
Смотрите раздел "Анализ устойчивости и сравнение с SUMO (eidm_stability_analysis.py)".

## Параметры симуляции (для `run_circle_simulation.py`)

- `--num-vehicles`: Количество автомобилей.
- `--simulation-duration`: Длительность симуляции в секундах.
- `--sumo-binary`: Путь к `sumo-gui.exe` или `sumo.exe`.
- `--config-name`: Имя конфигурации (влияет на имя директории результатов).
- `--detector-spacing`: Расстояние между детекторами (м).
- `--detector-sampling-period`: Период опроса детекторов (с).
- Другие параметры IDM могут быть заданы как константы в скрипте или переданы через аргументы, если реализовано.

## Результаты

### Результаты симуляции
Результаты каждой симуляции, запущенной через `run_circle_simulation.py` или `eidm_stability_analysis.py`, сохраняются в уникальной поддиректории внутри `results/`. Эти данные включают:
- `fcd_output_*.csv`: Данные о траекториях транспортных средств (Floating Car Data).
- `V_data_detectors.npy`: Матрица средних скоростей с каждого детектора за каждый период опроса.
- `N_data_detectors.npy`: Матрица количества ТС, прошедших через каждый детектор за каждый период опроса.
- `Q_data_detectors.npy`: Матрица интенсивности потока (ТС/с) для каждого детектора.
- `rho_data_detectors.npy`: Матрица плотности потока (ТС/м) для каждого детектора.
- `cameras_coords.npy`: NumPy массив с координатами каждого детектора на кольце.
- `Ts_detector.txt`: Файл, содержащий значение периода опроса детекторов `Ts`.
- `rt_detected_wave_events.csv`: CSV файл с информацией о волнах stop-and-go, обнаруженных в реальном времени модулем `RealTimeWaveDetector` во время симуляции. Содержит `t_event_s`, `x_event_m`, `wave_speed_mps` и др.
- `additional.xml`: Файл с описанием детекторов, сгенерированный для SUMO.
- `*.sumocfg`, `*.rou.xml`: Конфигурационные файлы SUMO для данного запуска.

### Результаты анализа
Скрипт `analyze_circle_data.py` обрабатывает данные из директории результатов симуляции и создает поддиректорию `analysis_plots_YYYY`, содержащую:
- `spacetime_heatmap_fcd_rt_annotated.png`: Пространственно-временная тепловая карта скорости `V(x,t)`, построенная по FCD данным. На эту карту нанесены вертикальные пунктирные линии, отмечающие моменты времени обнаружения stop-and-go волн (`t_event_s`) из `rt_detected_wave_events.csv` (с учетом `warmup_time`).
- `rhot_profile_detectors_annotated.png`: График зависимости плотности `rho(t)` от времени для первого детектора, также с аннотациями `rt_wave_event_times`.
- Стандартные графики на основе FCD:
    - `velocity_time_all.png` (V(t) для всех ТС)
    - `distance_time_all.png` (S(t) для всех ТС)
    - `velocity_position_all.png` (V(x) для всех ТС)
    - `fft_mean_speed.png` (Амплитудный спектр Фурье средней скорости)
    - И другие варианты этих графиков (для выбранных ТС и т.д.)
- `analysis_summary.json`: Сводка анализа, включая метрику `waves_observed` (основанную на стандартном отклонении скорости из FCD) и другие параметры.

Функция `detect_stop_and_go_waves` в `eidm_stability_analysis.py` предназначена для более детального анализа данных с детекторов (включая N, V, Q, rho) с целью обнаружения волн и может генерировать файл `detected_wave_events_from_postprocessing.csv`. Однако, её интеграция для автоматической аннотации всех графиков в `analyze_circle_data.py` на данный момент ограничена (большая часть соответствующего кода закомментирована в пользу использования `rt_detected_wave_events.csv`).

## Анализ устойчивости и сравнение с SUMO (eidm_stability_analysis.py)

Скрипт `src/eidm_stability_analysis.py` предназначен для проведения теоретического анализа устойчивости транспортного потока, описываемого моделью IDM (Intelligent Driver Model), и опционального сравнения теоретических предсказаний с результатами симуляций в SUMO для кольцевой дороги.

### Запуск

```bash
# Только теоретический анализ
python src/eidm_stability_analysis.py

# Теоретический анализ + запуск симуляций SUMO для сравнения
python src/eidm_stability_analysis.py --run-sumo-simulations --sumo-binary /path/to/sumo-gui.exe --sumo-tools-dir /path/to/sumo/tools

# Теоретический анализ + запуск симуляций SUMO с VSL контроллером
python src/eidm_stability_analysis.py --run-sumo-simulations --vsl --sumo-binary /path/to/sumo-gui.exe --sumo-tools-dir /path/to/sumo/tools
```

### Функционал

1.  **Теоретический анализ:**
    *   Находит равновесные состояния (скорость `v_e`, чистый зазор `s_e_net`) для различных условий (например, для заданного потока `Q`).
    *   Вычисляет частные производные IDM (`f_s`, `f_dv`, `f_v`) в точках равновесия.
    *   Определяет устойчивость взвода (platoon stability).
    *   Определяет устойчивость потока (string stability) с помощью критерия `K = f_v²/2 - f_dv*f_v - f_s`.
    *   Генерирует графики, иллюстрирующие:
        *   Фундаментальную диаграмму (v* от s*_net).
        *   Зависимость производных и критерия K от скорости.
        *   Области устойчивости взвода и потока.
        *   Влияние параметров IDM (например, времени реакции `T`) на устойчивость при фиксированных условиях (например, потоке `Q`).

2.  **Интеграция с SUMO (при флаге `--run-sumo-simulations`):**
    *   Для выбранного сценария (например, варьирование `T` при фиксированном `Q`):
        *   Для каждого теоретического равновесного состояния (`v_e`, `s_e_net`) рассчитывает необходимое количество машин `N` для кольца фиксированной длины.
        *   **Вызывает `src/run_circle_simulation.py`:** Запускает симуляцию SUMO для каждого рассчитанного `N`.
            *   **Передача параметров:** `eidm_stability_analysis.py` передает в `run_circle_simulation.py` рассчитанное количество транспортных средств (`--num-vehicles`), длительность симуляции, путь к SUMO, а также флаг `--vsl`, если он был указан при запуске `eidm_stability_analysis.py`.
            *   **Подготовка конфигурации:** `run_circle_simulation.py` генерирует или модифицирует файлы конфигурации SUMO:
                *   `*.rou.xml`: Файл маршрутов, создается на основе `--num-vehicles` и параметров кольцевой дороги.
                *   `additional.xml`: Файл с описанием индукционных детекторов (E2), расставленных на кольце согласно параметрам `--detector-spacing` и `--detector-sampling-period`.
                *   `*.sumocfg`: Главный конфигурационный файл SUMO, объединяющий сетевой файл, файл маршрутов и файл дополнительных элементов.
            *   **Инициализация TraCI и компонентов:** Скрипт устанавливает соединение с SUMO через TraCI.
                *   **VSL Controller (если `--vsl` активен):**
                    *   `run_circle_simulation.py` загружает конфигурацию для VSL из `config/vsl_config.yml` (если путь не переопределен).
                    *   Инициализируется экземпляр `VSLController` из `src/vsl_controller.py`. Контроллер получает доступ к TraCI для управления симуляцией и считывания данных.
                *   **RealTimeWaveDetector:** Инициализируется экземпляр `RealTimeWaveDetector` из `src/traci_interaction.py`. Этот детектор подключается к информации от индукционных петель, настроенных в `additional.xml`.
            *   **Симуляционный цикл:**
                *   На каждом шаге симуляции (`traci.simulationStep()`):
                    *   `RealTimeWaveDetector` собирает данные о скорости, количестве ТС и занятости с детекторов. Он анализирует эти данные в реальном времени для обнаружения событий волн "stop-and-go". Обнаруженные события (время, место, скорость волны и т.д.) записываются в файл `rt_detected_wave_events.csv` в директории результатов текущего запуска.
                    *   **Если VSL контроллер активен (`vsl_controller.is_enabled()`):**
                        *   `vsl_controller.step(simulation_time)` вызывается на каждом шаге (или согласно его внутреннему интервалу `Ts`).
                        *   Контроллер анализирует текущее состояние потока (например, плотность, рассчитанную по данным с одного из детекторов, указанного в его конфигурации).
                        *   На основе ПИД-регулятора, контроллер рассчитывает новое целевое ограничение скорости.
                        *   **Управление скоростью:** Рассчитанное ограничение скорости применяется ко всем **полосам** (`lane_ids`), указанным в конфигурации контроллера, с помощью команды TraCI `lane.setAllowedSpeed()`.
                        *   **Визуальные знаки:** Одновременно с детекторами генерируются XML-элементы `variableSpeedSign` (по одному на каждую полосу кольца) в файле `detectors.add.xml`. Эти знаки служат для **визуального отображения** в SUMO GUI, но фактическое управление скоростью происходит через изменение атрибута разрешенной скорости полосы. Скорость на этих визуальных знаках может не обновляться динамически через TraCI, если это не поддерживается версией SUMO или имеет ограничения.
                        *   Данные о работе VSL (целевая и текущая плотность, ошибка, выход ПИД-регулятора, установленная скорость) логируются в CSV файл (например, `vsl_log.csv`) в директории результатов.
            *   **Сохранение результатов по завершении симуляции:**
                *   Данные FCD (Floating Car Data) сохраняются в `fcd_output_*.csv`.
                *   Агрегированные данные с детекторов (средние скорости `V_data_detectors.npy`, количество ТС `N_data_detectors.npy`, поток `Q_data_detectors.npy`, плотность `rho_data_detectors.npy`), а также координаты детекторов (`cameras_coords.npy`) и период их опроса (`Ts_detector.txt`) сохраняются в соответствующие файлы.
        *   **Вызывает `src/analyze_circle_data.py`:** Анализирует полученные FCD данные и данные детекторов из директории результатов. Этот скрипт:
            *   Вычисляет метрики (например, стандартное отклонение скорости из FCD для флага `waves_observed`).
            *   Строит графики (см. раздел "Результаты анализа"), включая FCD тепловую карту и `rho(t)` с аннотациями из `rt_detected_wave_events.csv`.
            *   Сохраняет сводку анализа в `analysis_summary.json`.
            *   **Считывает `analysis_summary.json`:** Извлекает результат симуляции (`waves_observed` на основе FCD).
        *   **Сравнение:** Перерисовывает теоретический график устойчивости (например, T vs Устойчивость потока), добавляя на него точки, полученные из симуляций SUMO, помеченные как стабильные или нестабильные (на основе `waves_observed` из FCD). Данные из `rt_detected_wave_events.csv` дают дополнительное, более прямое подтверждение наличия волн, которое отображается на графиках анализа. Работа VSL контроллера и его влияние на стабильность потока будут отражены в результатах симуляции и могут быть сопоставлены с теоретическими предсказаниями.
