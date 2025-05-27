# Структура и зависимости кодовой базы

Этот документ описывает ключевые компоненты проекта и их взаимодействие, фокусируясь на двух основных сценариях использования.

## Основные компоненты

- **`src/eidm_stability_analysis.py`**: Отвечает за теоретический анализ устойчивости модели IDM и запуск серий симуляций в SUMO для валидации. Это основной скрипт для двух сценариев использования.
  - Детальное описание: [Документация по eidm_stability_analysis](.supercode/docs/eidm_stability_analysis_docs.md)
- **`src/run_circle_simulation.py`**: Запускает одиночные симуляции SUMO на кольцевой трассе, управляет сбором данных с детекторов и FCD, а также может активировать VSL-контроллер. Вызывается из `eidm_stability_analysis.py`.
  - Детальное описание: [Документация по run_circle_simulation](.supercode/docs/run_circle_simulation_docs.md)
- **`src/analyze_circle_data.py`**: Анализирует результаты симуляций (FCD и данные детекторов), генерирует графики и сводку. Вызывается из `eidm_stability_analysis.py` после завершения симуляций.
  - Детальное описание: [Документация по analyze_circle_data](.supercode/docs/analyze_circle_data_docs.md)
- **`src/traci_interaction.py`**: Модуль для инкапсуляции взаимодействия с SUMO через TraCI, включая запуск/остановку симуляции и сбор данных с детекторов. Также содержит класс `RealTimeWaveDetector` для обнаружения волн в реальном времени. Используется `run_circle_simulation.py`.
  - Детальное описание: [Документация по traci_interaction](.supercode/docs/traci_interaction_docs.md)
- **`src/vsl_controller.py`**: Реализует логику VSL-контроллера (управление переменными скоростными ограничениями). Используется `run_circle_simulation.py` при активации VSL.
  - Детальное описание: [Документация по vsl_controller](.supercode/docs/vsl_controller_docs.md)
- **`src/theory_breakdown.py`**: Скрипт для теоретического анализа образования пробок. Кажется, он менее интегрирован в основной пайплайн, но полезен для понимания теоретических аспектов.
  - Детальное описание: [Документация по theory_breakdown](.supercode/docs/theory_breakdown_docs.md)
- **`README.md`**: Общее описание проекта, инструкции по установке и использованию.
- **`THEORETICAL_BACKGROUND.md`**: Детальное описание теоретических основ, используемых в проекте.

## Сценарии использования

### 1. Анализ устойчивости и симуляция без VSL

**Команда:**
```bash
python src/eidm_stability_analysis.py --run-sumo-simulations
```

**Поток выполнения:**
1.  **`eidm_stability_analysis.py`**:
    *   Проводит теоретический анализ устойчивости IDM.
    *   Для набора параметров (например, варьируя время реакции `T` при фиксированном потоке `Q`) определяет равновесные состояния (`v_e`, `s_e_net`).
    *   Для каждого равновесного состояния:
        *   Вызывает **`run_circle_simulation.py`**, передавая рассчитанное количество машин и другие параметры. VSL **не** активируется.
2.  **`run_circle_simulation.py`**:
    *   Генерирует конфигурационные файлы для SUMO (маршруты, детекторы).
    *   Запускает симуляцию SUMO.
    *   Использует **`traci_interaction.py`** для:
        *   Управления симуляцией.
        *   Сбора данных с детекторов E1 (скорость, количество ТС).
        *   Работы `RealTimeWaveDetector` для обнаружения волн в реальном времени и сохранения `rt_detected_wave_events.csv`.
    *   По завершении сохраняет FCD данные и агрегированные данные детекторов.
3.  **`eidm_stability_analysis.py`** (продолжение):
    *   После завершения каждой симуляции вызывает **`analyze_circle_data.py`**.
4.  **`analyze_circle_data.py`**:
    *   Загружает FCD данные и данные детекторов из директории результатов `run_circle_simulation.py`.
    *   Загружает `rt_detected_wave_events.csv`.
    *   Генерирует графики (тепловые карты, V(t), rho(t) и т.д.), аннотируя их информацией об обнаруженных волнах.
    *   Сохраняет `analysis_summary.json` с ключевыми метриками (например, `waves_observed`).
5.  **`eidm_stability_analysis.py`** (завершение):
    *   Собирает результаты анализа (`waves_observed`) от `analyze_circle_data.py`.
    *   Обновляет теоретические графики устойчивости, добавляя точки, соответствующие результатам симуляций.

### 2. Анализ устойчивости и симуляция с VSL

**Команда:**
```bash
python src/eidm_stability_analysis.py --run-sumo-simulations --vsl
```

**Поток выполнения:**
Отличается от сценария 1 в следующих моментах:

1.  **`eidm_stability_analysis.py`**:
    *   При вызове `run_circle_simulation.py` передает дополнительный флаг или параметр для активации VSL.
2.  **`run_circle_simulation.py`**:
    *   Распознает флаг активации VSL.
    *   Инициализирует и использует **`vsl_controller.py`**.
    *   `VSLController`:
        *   Взаимодействует с симуляцией через **`traci_interaction.py`** (или напрямую через переданный `traci_conn`).
        *   Считывает данные с указанного в его конфигурации детектора.
        *   Рассчитывает и применяет новые ограничения скорости к указанным полосам.
        *   Логирует свою работу в `vsl_log.csv`.
    *   `RealTimeWaveDetector` в `traci_interaction.py` работает параллельно, как и в сценарии без VSL.
3.  **`analyze_circle_data.py`** и последующие шаги в `eidm_stability_analysis.py` аналогичны сценарию 1, но результаты симуляций теперь отражают работу VSL. `vsl_log.csv` также доступен для анализа влияния VSL.

## Ключевые зависимости между файлами

-   `eidm_stability_analysis.py` -> `run_circle_simulation.py`, `analyze_circle_data.py`, `eidm_stability_analysis` (импорт функций из самого себя для теоретических расчетов).
-   `run_circle_simulation.py` -> `traci_interaction.py`, `vsl_controller.py` (опционально).
-   `analyze_circle_data.py` -> `eidm_stability_analysis` (для импорта `detect_stop_and_go_waves`, хотя она может быть закомментирована).
-   `traci_interaction.py`: Содержит `RealTimeWaveDetector`, самостоятельный.
-   `vsl_controller.py`: Самостоятельная логика VSL, принимает `traci_conn`.

Эта структура позволяет модульно изменять компоненты. Например, улучшение алгоритма VSL затрагивает в основном `vsl_controller.py` и, возможно, способ его вызова/конфигурирования в `run_circle_simulation.py`. 