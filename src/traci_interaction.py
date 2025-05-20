import os
import sys
import traci
import libsumo 
import numpy as np
import pandas as pd
from scipy.signal import correlate as scipy_correlate, find_peaks
from collections import deque

# Модуль для инкапсуляции взаимодействия с SUMO/TraCI

SUMO_BINARY_GUI = "sumo-gui"
SUMO_BINARY_CLI = "sumo"

# Параметры по умолчанию для RealTimeWaveDetector
# Их можно переопределить при создании экземпляра класса
RT_CORR_THRESH = 0.85  # Порог для коэффициента корреляции
RT_MOVING_AVG_WINDOW = 5  # Окно для скользящего среднего (в шагах Ts)
RT_MIN_SPEED_DROP_FACTOR = 0.5  # Фактор для определения нижней границы падения скорости
RT_MIN_SPEED_FOR_DROP_DETECTION_FACTOR = 0.7  # Фактор для определения скорости перед падением
RT_BUFFER_SIZE_SECONDS = 60  # Сколько секунд данных хранить в буфере для анализа
RT_MIN_COOLDOWN_SECONDS = 30 # Минимальный интервал в секундах перед повторным сообщением о волне для той же пары камер


class RealTimeWaveDetector:
    def __init__(self, detector_ids_ordered, cameras_coords, Ts,
                 buffer_size_seconds=RT_BUFFER_SIZE_SECONDS,
                 mov_avg_window=RT_MOVING_AVG_WINDOW,
                 corr_thresh=RT_CORR_THRESH,
                 min_speed_drop_factor=RT_MIN_SPEED_DROP_FACTOR,
                 min_speed_for_drop_detection_factor=RT_MIN_SPEED_FOR_DROP_DETECTION_FACTOR,
                 min_cooldown_seconds=RT_MIN_COOLDOWN_SECONDS):
        self.detector_ids = detector_ids_ordered
        self.cameras_coords = np.array(cameras_coords)
        self.Ts = Ts
        
        self.mov_avg_window = mov_avg_window
        # Минимальное количество точек в буфере должно быть достаточным для работы скользящего среднего
        min_buffer_steps_for_filter = self.mov_avg_window * 2 
        self.buffer_size_steps = max(int(buffer_size_seconds / Ts), min_buffer_steps_for_filter)
        
        if int(buffer_size_seconds / Ts) < min_buffer_steps_for_filter:
            print(f"TRACI_INTERACTION - RealTimeWaveDetector: INFO: buffer_size_seconds ({buffer_size_seconds}s) "
                  f"слишком мал для окна сглаживания {self.mov_avg_window} (требуется {min_buffer_steps_for_filter*self.Ts}s). "
                  f"Размер буфера увеличен до {self.buffer_size_steps * self.Ts:.1f}s ({self.buffer_size_steps} шагов).")

        self.corr_thresh = corr_thresh
        self.min_speed_drop_factor = min_speed_drop_factor
        self.min_speed_for_drop_detection_factor = min_speed_for_drop_detection_factor
        self.min_cooldown_seconds = min_cooldown_seconds

        self.num_cameras = len(self.detector_ids)
        self.v_data_buffers = [deque(maxlen=self.buffer_size_steps) for _ in range(self.num_cameras)]
        self.n_data_buffers = [deque(maxlen=self.buffer_size_steps) for _ in range(self.num_cameras)] # Новый буфер для N
        self.q_data_buffers = [deque(maxlen=self.buffer_size_steps) for _ in range(self.num_cameras)] # Новый буфер для Q
        self.rho_data_buffers = [deque(maxlen=self.buffer_size_steps) for _ in range(self.num_cameras)] # Новый буфер для rho
        
        self.detector_id_to_idx = {det_id: i for i, det_id in enumerate(self.detector_ids)}
        self.last_wave_report_time_s_per_pair = {} # Ключ: (cam_idx_1, cam_idx_2), Значение: last_t_event_s
        self.detected_event_details_list = [] # Новый список для хранения деталей событий

    def update_and_detect(self, current_sim_time_s, new_readings_dict):
        """
        Обновляет буферы данных с детекторов и выполняет обнаружение волн.
        Выводит информацию об обнаруженных волнах в консоль.

        Args:
            current_sim_time_s (float): Текущее время симуляции в секундах.
            new_readings_dict (dict): Словарь с последними данными от детекторов.
                                      Формат: {det_id: {'vehicle_count': int, 'mean_speed': float}}
        """
        # 1. Обновление буферов
        for det_id, data in new_readings_dict.items():
            if det_id in self.detector_id_to_idx:
                cam_idx = self.detector_id_to_idx[det_id]
                # Убедимся, что скорость не NaN перед добавлением, заменим на 0, если так
                speed_to_add = data.get('mean_speed', 0.0)
                if pd.isna(speed_to_add) or speed_to_add < 0: # Скорость -1 от TraCI означает нет машин
                    speed_to_add = 0.0 
                self.v_data_buffers[cam_idx].append(speed_to_add)

                count_to_add = data.get('vehicle_count', 0)
                if pd.isna(count_to_add) or count_to_add < 0:
                    count_to_add = 0
                self.n_data_buffers[cam_idx].append(float(count_to_add))

                # Расчет Q и rho для текущего шага и добавление в буферы
                current_q = float(count_to_add) / self.Ts if self.Ts > 1e-9 else 0.0
                self.q_data_buffers[cam_idx].append(current_q)

                current_rho = 0.0
                if speed_to_add > 1e-3: # Избегаем деления на ноль или очень малую скорость
                    current_rho = current_q / speed_to_add
                elif current_q > 1e-3: # Если есть поток, но скорость почти ноль, плотность будет высокой
                    current_rho = np.nan # или очень большое число, но nan лучше для дальнейшего анализа, если он будет
                self.rho_data_buffers[cam_idx].append(current_rho)
            #else:
            #    print(f"TRACI_INTERACTION - RealTimeWaveDetector: WARNING: Detector ID {det_id} from new_readings_dict not in configured detector_ids.")
        
        if self.num_cameras <= 1:
            return

        # 2. Обнаружение волн для каждой пары соседних камер
        for i in range(self.num_cameras - 1):
            cam_idx_1 = i
            cam_idx_2 = i + 1

            if len(self.v_data_buffers[cam_idx_1]) < self.buffer_size_steps or \
               len(self.v_data_buffers[cam_idx_2]) < self.buffer_size_steps:
                continue # Недостаточно данных в буфере для анализа этой пары

            x_i = self.cameras_coords[cam_idx_1]
            x_i_plus_1 = self.cameras_coords[cam_idx_2]
            delta_x = x_i_plus_1 - x_i
            if abs(delta_x) < 1e-3:
                continue

            vbar_i_raw = np.array(self.v_data_buffers[cam_idx_1])
            vbar_i_plus_1_raw = np.array(self.v_data_buffers[cam_idx_2])

            vbar_i_filt = pd.Series(vbar_i_raw).rolling(window=self.mov_avg_window, center=True, min_periods=1).mean().to_numpy()
            vbar_i_plus_1_filt = pd.Series(vbar_i_plus_1_raw).rolling(window=self.mov_avg_window, center=True, min_periods=1).mean().to_numpy()

            valid_indices = ~np.isnan(vbar_i_filt) & ~np.isnan(vbar_i_plus_1_filt)
            if np.sum(valid_indices) < self.mov_avg_window: # Проверка после фильтрации
                continue
            
            v1 = vbar_i_filt[valid_indices]
            v2 = vbar_i_plus_1_filt[valid_indices]
            
            if len(v1) < self.mov_avg_window or len(v2) < self.mov_avg_window : # Еще одна проверка на длину
                continue

            mean_v1, mean_v2 = np.mean(v1), np.mean(v2)
            v1_centered = v1 - mean_v1
            v2_centered = v2 - mean_v2
            
            # Избегаем корреляции, если один из сигналов имеет нулевую дисперсию (константа)
            if np.sum(v1_centered**2) < 1e-9 or np.sum(v2_centered**2) < 1e-9:
                continue

            correlation = scipy_correlate(v1_centered, v2_centered, mode='full')
            norm_factor = np.sqrt(np.sum(v1_centered**2) * np.sum(v2_centered**2))
            if norm_factor < 1e-9:
                continue # Нормализационный фактор слишком мал
            
            normalized_correlation = correlation / norm_factor
            
            len_v_corr = len(v1) 
            lags = np.arange(-(len_v_corr - 1), len_v_corr)

            positive_lags_indices = np.where(lags > 0)[0]
            if not positive_lags_indices.any():
                continue
            
            R_positive_lags = normalized_correlation[positive_lags_indices]
            if not R_positive_lags.any():
                continue

            peaks_indices_in_R_positive, properties = find_peaks(R_positive_lags, height=self.corr_thresh)
            if not peaks_indices_in_R_positive.any():
                continue

            idx_of_max_peak_in_R_positive = peaks_indices_in_R_positive[np.argmax(properties['peak_heights'])]
            R_max_val = properties['peak_heights'][np.argmax(properties['peak_heights'])]
            tau_max_steps_in_lags = lags[positive_lags_indices[idx_of_max_peak_in_R_positive]]
            tau_max_time = tau_max_steps_in_lags * self.Ts

            if abs(tau_max_time) < 1e-9: wave_speed = float('inf') if delta_x != 0 else float('nan')
            else: wave_speed = -delta_x / tau_max_time

            mean_speed_on_cam_i_buffer = np.mean(vbar_i_filt[~np.isnan(vbar_i_filt)])
            if np.isnan(mean_speed_on_cam_i_buffer): continue

            speed_thresh_low = self.min_speed_drop_factor * mean_speed_on_cam_i_buffer
            speed_thresh_high = self.min_speed_for_drop_detection_factor * mean_speed_on_cam_i_buffer
            if speed_thresh_high < speed_thresh_low : # Гарантируем, что высокий порог не ниже низкого
                speed_thresh_high = speed_thresh_low + 1e-3

            k_peak_in_buffer = -1
            was_above_high = False
            for k_buf in range(len(vbar_i_filt)):
                if np.isnan(vbar_i_filt[k_buf]): continue
                if was_above_high and vbar_i_filt[k_buf] < speed_thresh_low:
                    k_peak_in_buffer = k_buf
                    break 
                if vbar_i_filt[k_buf] > speed_thresh_high: was_above_high = True
                elif vbar_i_filt[k_buf] < speed_thresh_low: was_above_high = False
            
            if k_peak_in_buffer == -1: continue

            # Время события: t_event_s - это абсолютное время симуляции
            # k_peak_in_buffer - индекс в буфере (0 до buffer_size_steps-1)
            # Последний элемент буфера соответствует current_sim_time_s
            t_event_s_candidate = current_sim_time_s - (self.buffer_size_steps - 1 - k_peak_in_buffer) * self.Ts
            x_event_m = self.cameras_coords[cam_idx_1]

            pair_key = (cam_idx_1, cam_idx_2)
            last_reported_t = self.last_wave_report_time_s_per_pair.get(pair_key, -float('inf'))
            
            if t_event_s_candidate > last_reported_t + self.min_cooldown_seconds:
                self.last_wave_report_time_s_per_pair[pair_key] = t_event_s_candidate
                
                # Собираем данные о событии для CSV и, возможно, для возврата
                event_details = {
                    't_event_s': t_event_s_candidate,
                    'sim_time_at_detection_s': current_sim_time_s,
                    'x_event_m': x_event_m,
                    'cam_idx_1': cam_idx_1,
                    'detector_id_1': self.detector_ids[cam_idx_1],
                    'cam_idx_2': cam_idx_2,
                    'detector_id_2': self.detector_ids[cam_idx_2],
                    'dx_m': delta_x,
                    'dt_sync_s': tau_max_time,
                    'wave_speed_mps': wave_speed,
                    'R_max_corr': R_max_val
                }
                self.detected_event_details_list.append(event_details) # Сохраняем детали

                # Новый подробный вывод (ЗАКОММЕНТИРОВАНО):
                # print(f"\n[RealTimeWaveDetector] ОБНАРУЖЕНА ВОЛНА STOP-AND-GO:")
                # print(f"  Время события (t_event): {t_event_s_candidate:.2f} с (симуляционное время: {current_sim_time_s:.2f} c)")
                # print(f"  Позиция события (x_event): {x_event_m:.2f} м (на камере {cam_idx_1}: {self.detector_ids[cam_idx_1]})")
                # print(f"  Пара камер: {cam_idx_1} ({self.detector_ids[cam_idx_1]}) -> {cam_idx_2} ({self.detector_ids[cam_idx_2]})")
                # print(f"  Расстояние между камерами (dx): {delta_x:.2f} м")
                # print(f"  Временной лаг (dt_sync): {tau_max_time:.2f} с (шагов: {tau_max_time/self.Ts:.1f})")
                # print(f"  Скорость волны (c): {wave_speed:.2f} м/с")
                # print(f"  Макс. нормированная корреляция (R_max): {R_max_val:.3f}")
                pass # Оставляем pass, если после if больше ничего нет

    def get_detected_event_details(self):
        """Возвращает список всех деталей обнаруженных событий."""
        return self.detected_event_details_list

def connect_sumo(sumo_cmd_list, port=None):
    """
    Запускает SUMO как подпроцесс и устанавливает TraCI соединение.

    Args:
        sumo_cmd_list (list): Список аргументов для запуска SUMO (например, [sumo_binary, "-c", config_file]).
        port (int, optional): Явный порт для TraCI. Если None, TraCI выберет свободный.

    Returns:
        bool: True в случае успеха, False иначе.
    """
    try:
        if port:
            traci.init(port)
        else:
            # Для автоматического выбора порта и запуска SUMO
            # traci.start использует sumo_cmd_list напрямую
            traci.start(sumo_cmd_list)
        print(f"TraCI: Успешно подключено к SUMO. Версия TraCI: {traci.getVersion()}")
        return True
    except traci.TraCIException as e:
        print(f"TraCI: Ошибка подключения/запуска SUMO: {e}")
        # Попытка убить процесс SUMO, если он был запущен traci.start и завис
        if hasattr(traci, '_connections') and 'default' in traci._connections:
            conn = traci._connections['default']
            if conn._sumoProcess:
                try:
                    conn._sumoProcess.kill()
                except Exception as kill_e:
                    print(f"Ошибка при попытке остановить процесс SUMO: {kill_e}")
        return False
    except Exception as e: # Другие возможные ошибки (например, sumo_cmd_list некорректен)
        print(f"Непредвиденная ошибка при запуске/подключении к SUMO: {e}")
        return False

def simulation_step(time_step_ms=None):
    """
    Выполняет один шаг симуляции в SUMO.

    Args:
        time_step_ms (int, optional): Длительность шага симуляции в миллисекундах.
                                     Если None, используется шаг по умолчанию из SUMO.
                                     Если указано, traci.simulationStep(targetTime) будет использован.
                                     Обратите внимание: если вы передаете time_step_ms, это должно быть
                                     общее время, до которого нужно симулировать, а не дельта.
                                     Для простого шага вперед используйте traci.simulationStep() без аргументов.
    """
    try:
        if time_step_ms is not None:
            current_time_ms = traci.simulation.getTime()
            traci.simulationStep(current_time_ms + time_step_ms)
        else:
            traci.simulationStep()
    except traci.TraCIException as e:
        print(f"TraCI: Ошибка во время шага симуляции: {e}")
        # Здесь может потребоваться более сложная обработка ошибок,
        # например, попытка закрыть соединение.
        raise # Перевыбрасываем исключение, чтобы вызывающий код мог его обработать

def get_detector_data(detector_ids):
    """
    Собирает данные (число машин, средняя скорость) с указанных детекторов E1.

    Args:
        detector_ids (list[str]): Список ID детекторов (E1 induction loops).

    Returns:
        dict: Словарь, где ключи - ID детекторов, а значения - словари
              {'vehicle_count': int, 'mean_speed': float}.
              'mean_speed' будет -1, если машины не проезжали или данные недоступны.
    """
    detector_results = {}
    for det_id in detector_ids:
        try:
            # Данные за ПОСЛЕДНИЙ завершенный интервал времени (согласно freq детектора)
            vehicle_count = traci.inductionloop.getLastStepVehicleNumber(det_id)
            mean_speed = traci.inductionloop.getLastStepMeanSpeed(det_id) # м/с
            
            # getLastStepMeanSpeed возвращает -1, если нет машин. Это нормально.
            detector_results[det_id] = {
                'vehicle_count': vehicle_count,
                'mean_speed': mean_speed 
            }
        except traci.TraCIException as e:
            print(f"TraCI: Ошибка при получении данных с детектора {det_id}: {e}")
            detector_results[det_id] = {
                'vehicle_count': 0, # или None, или обработать ошибку иначе
                'mean_speed': -1.0   # или None
            }
        except Exception as e: # Другие возможные ошибки
            print(f"Непредвиденная ошибка при запросе данных детектора {det_id}: {e}")
            detector_results[det_id] = {
                'vehicle_count': 0,
                'mean_speed': -1.0
            }
    return detector_results

def close_sumo():
    """
    Закрывает TraCI соединение с SUMO.
    """
    try:
        traci.close()
        print("TraCI: Соединение с SUMO закрыто.")
    except traci.TraCIException as e:
        print(f"TraCI: Ошибка при закрытии соединения: {e}")
    except Exception as e: # Если traci не был инициализирован
        print(f"Ошибка при вызове traci.close(): {e}")

# Пример использования (можно раскомментировать для теста, если SUMO и конфиг доступны)
if __name__ == '__main__':
    print("Модуль traci_interaction.py загружен.")
    # Для реального использования этот блок if __name__ == '__main__' не нужен,
    # функции будут импортироваться в run_circle_simulation.py 