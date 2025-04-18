# AI Circle Simulation

Проект для моделирования движения транспортных средств по круговой дороге с использованием SUMO (Simulation of Urban MObility) и EIDM (Enhanced Intelligent Driver Model).

## Структура проекта

```
.
├── config/
│   ├── network/     # Файлы сети (circle.net.xml, circle.nod.xml, circle.edg.xml)
│   └── routes/      # Файлы маршрутов (генерируются автоматически)
├── data/
│   └── simulations/ # Результаты симуляций
├── results/
│   └── analysis/    # Результаты анализа данных
├── src/
│   ├── generate_circle_rou_new.py  # Генератор маршрутов
│   ├── run_circle.py              # Скрипт запуска симуляции
│   └── analyze_data.py           # Скрипт анализа данных
└── requirements.txt # Зависимости проекта
```

## Установка

1. Установите SUMO (Simulation of Urban MObility):
   - Windows: Скачайте и установите с [официального сайта](https://sumo.dlr.de/docs/Installing/Windows.html)
   - Linux: `sudo apt-get install sumo`

2. Установите зависимости проекта:
   ```bash
   pip install -r requirements.txt
   ```

## Использование

1. Генерация маршрутов:
   ```bash
   cd src
   python generate_circle_rou_new.py
   ```

2. Запуск симуляции (включает автоматический анализ данных):
   ```bash
   cd src
   python run_circle.py
   ```

3. Анализ данных (опционально, запускается автоматически после симуляции):
   ```bash
   cd src
   python analyze_data.py --file путь/к/файлу/с/данными.csv
   ```

## Параметры симуляции

- Количество машин на каждом маршруте: 7
- Длина ребра: 87.15 метров
- Параметры EIDM:
  - Ускорение: 2.6 ± 0.5 м/с²
  - Замедление: 4.5 ± 0.5 м/с²
  - Сигма: 0.5 ± 0.1
  - Тау: 0.5 ± 0.1
  - Дельта: 0.5 ± 0.1
  - Степпинг: 0.5 ± 0.1

## Результаты

### Результаты симуляции
Результаты симуляции сохраняются в директории `data/simulations/` в формате CSV. 

### Результаты анализа
После завершения симуляции автоматически создаются следующие графики в директории `results/analysis/`:
- `velocity_time.png` - график скорости от времени V(t)
- `distance_time.png` - график пройденного расстояния от времени S(t)

На графиках:
- Красной линией выделен тормозящий автомобиль
- Синим цветом показаны остальные автомобили
- Вертикальной пунктирной линией отмечен момент торможения (10с) 