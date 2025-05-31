#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vsl_controller.py

Полный одноклавишный скрипт VSL‑управления для SUMO.

Теория:

1. Линеаризация IDM (Intelligent Driver Model) вокруг точки равновесия (v_e, s_e):
   Функция ускорения IDM:
   a_IDM = a * (1 - (v/v0)^delta - ((s0 + v*T + v*(v-v_prev)/(2*sqrt(a*b)))/s)^2)

   Для упрощения и линеаризации часто рассматривают стационарное состояние (v_prev = v),
   и s_e - равновесное расстояние, соответствующее скорости v_e.
   s_e(v_e) = s0 + v_e * T  (если пренебречь членом sqrt(ab) для простоты или когда v_e=0)

   Парциальные производные функции ускорения (f = dv/dt) по состоянию (s, v) и параметру v0 (желаемая скорость):
   Предположим, равновесное состояние s_e соответствует v_e, где ускорение равно нулю.
   Рассматривается отклик системы на малые отклонения от (v_e, s_e) и изменения v0.

   f_s = ∂(dv/dt)/∂s  (влияние изменения расстояния на ускорение)
       = 2*a/s_e * (1 - (v_e/v0)^delta)  (при s_e = s0 + v_e*T)
         Более точная формула, если s* = s0 + vT + v(v-v_prev)/(2*sqrt(a*b)) используется в IDM,
         будет сложнее. Для фундаментальной диаграммы, s_e часто берут как 1/ρ_e.

   f_v = ∂(dv/dt)/∂v  (влияние изменения скорости на ускорение)
       = -a * [ delta * (v_e/v0)^(delta-1) / v0 + 2*T/s_e * (1 - (v_e/v0)^delta) ]
         (при s_e = s0 + v_e*T)

   f_v0 = ∂(dv/dt)/∂v0 (влияние изменения желаемой скорости v0 на ускорение)
        = a * delta * (v_e/v0)^delta / v0

   Эти коэффициенты используются для описания динамики потока вблизи равновесия.

2. Передаточная функция изменения табло (v0_VSL) → изменения скорости потока (v):
   Для гомогенного потока, линеаризованная модель поведения одной машины может быть расширена
   на поведение потока. Передаточная функция от изменения v0 (желаемой скорости, заданной VSL)
   к средней скорости потока v может быть выражена как:
   G(s) = f_v0 * s / (s^2 - f_v * s + f_s)
   Это модель второго порядка, описывающая, как изменения в v0 распространяются и влияют на v.
   Знаменатель определяет стабильность и отклик системы.

3. PI-регулятор (инкрементная форма):
   Цель: поддерживать плотность ρ на уровне ρ_crit.
   Ошибка: e[k] = ρ[k] - ρ_crit
   Изменение управляющего воздействия (Δu[k]):
   Δu[k] = u[k] - u[k-1]
         = Kp * (e[k] - e[k-1])                  (Пропорциональная часть)
           + Ki * Ts * e[k]                       (Интегральная часть)
           + Kd * (e[k] - 2*e[k-1] + e[k-2]) / Ts  (Дифференциальная часть)

   Новое значение желаемой скорости (ограничение скорости на табло):
   v_VSL[k] = v_initial_target - u[k]
   где u[k] = u[k-1] + Δu[k].
   v_VSL должна быть ограничена снизу (v_min) и сверху (изначальным v0).
   Обычно, v_VSL устанавливается как целевая v0 для IDM на контролируемом участке.
"""

import os
import sys
import logging
import time
import math
import collections
import csv
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Проверяем наличие переменной окружения SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    # Проверяем, не добавлен ли уже путь, чтобы избежать дублирования
    if tools not in sys.path:
        sys.path.append(tools)
else:
    # Если VSL Controller используется как модуль в более крупном проекте,
    # основное приложение должно позаботиться о SUMO_HOME и sys.path.
    pass 

import traci
from traci.exceptions import TraCIException

# НОВОЕ: Импорт VSLOptimizer для динамического расчета параметров
try:
    from vsl_calculator import VSLOptimizer, VSLParams, get_vsl_params_for_scenario
    VSL_CALCULATOR_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("VSLOptimizer недоступен. Используется статическая конфигурация.")
    VSL_CALCULATOR_AVAILABLE = False

# Настройка логирования
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Конфигурация логирования будет управляться из основного скрипта

DEFAULT_LOG_FILE_BASENAME = "vsl_log.csv"


# --- VSL Контроллер ---
class VSLController:
    """
    Контроллер для управления переменными скоростными ограничениями (VSL) в SUMO через TraCI.

    Поддерживает два режима работы:
    1. Статический - с фиксированными PID параметрами (обратная совместимость)
    2. Динамический - с математически обоснованными сценариями через VSLOptimizer

    Теоретическое обоснование (см. docstring модуля)
    """
    
    @classmethod
    def create_with_scenario(cls, 
                           traci_conn: Any,
                           scenario: str,
                           T_prime: float,
                           ctrl_segments_lanes: List[str],
                           vsl_detector_id: str,
                           vsl_detector_length_m: float,
                           log_csv_full_path: str,
                           sim_step_length: float,
                           ts_control_interval: float = 10.0,
                           idm_params: Optional[dict] = None,
                           stability_data: Optional[dict] = None,
                           enabled: bool = True):
        """
        Создать VSL контроллер с математически обоснованным сценарием.
        
        Args:
            traci_conn: Активное соединение TraCI
            scenario: Название сценария ('speed', 'variance', 'wave', 'throughput')
            T_prime: Время реакции водителя (0.5-1.5с)
            ctrl_segments_lanes: Список ID полос для управления VSL
            vsl_detector_id: ID детектора E1 для сбора данных
            vsl_detector_length_m: Длина зоны детектирования детектора (м)
            log_csv_full_path: Путь к файлу лога
            sim_step_length: Длительность шага симуляции (с)
            ts_control_interval: Интервал управления VSL (с)
            idm_params: Параметры IDM модели (опционально)
            stability_data: Данные анализа устойчивости (опционально)
            enabled: Включен ли контроллер
        """
        if not VSL_CALCULATOR_AVAILABLE:
            raise ImportError("VSLOptimizer недоступен. Невозможно создать контроллер со сценарием.")
        
        # Создаем VSL оптимизатор
        optimizer = VSLOptimizer(T_prime, idm_params)
        
        # Получаем параметры для сценария
        vsl_params = optimizer.get_scenario_params(scenario, stability_data)
        
        logger.info(f"Создание VSL контроллера со сценарием '{scenario}': "
                   f"Kp={vsl_params.kp:.3f}, Ki={vsl_params.ki:.3f}, Kd={vsl_params.kd:.3f}")
        
        # Создаем экземпляр контроллера
        controller = cls(
            traci_conn=traci_conn,
            idm_v0_default=optimizer.v0,  # Используем v0 из IDM параметров
            ctrl_segments_lanes=ctrl_segments_lanes,
            ts_control_interval=ts_control_interval,
            kp=vsl_params.kp,
            ki=vsl_params.ki,
            kd=vsl_params.kd,
            v_min_vsl_limit=vsl_params.vsl_bounds[0],
            rho_crit_target=vsl_params.target_density,
            vsl_detector_id=vsl_detector_id,
            vsl_detector_length_m=vsl_detector_length_m,
            log_csv_full_path=log_csv_full_path,
            sim_step_length=sim_step_length,
            enabled=enabled
        )
        
        # Сохраняем дополнительные параметры для динамического управления
        controller.vsl_optimizer = optimizer
        controller.scenario = scenario
        controller.vsl_params = vsl_params
        controller.v_max_vsl_limit = vsl_params.vsl_bounds[1]
        controller.use_dynamic_control = True
        
        logger.info(f"VSL контроллер создан для сценария '{scenario}' с T'={T_prime:.2f}с")
        logger.info(f"Целевая плотность: {vsl_params.target_density:.1f} авто/км, "
                   f"VSL диапазон: [{vsl_params.vsl_bounds[0]:.1f}, {vsl_params.vsl_bounds[1]:.1f}] м/с")
        
        return controller
    
    def __init__(self, 
                 traci_conn: Any, # Ожидается объект соединения traci
                 idm_v0_default: float,
                 ctrl_segments_lanes: List[str],
                 ts_control_interval: float,
                 kp: float, 
                 ki: float, 
                 kd: float,
                 v_min_vsl_limit: float,
                 rho_crit_target: float,
                 vsl_detector_id: str,
                 vsl_detector_length_m: float,
                 log_csv_full_path: str,
                 sim_step_length: float,
                 enabled: bool = True):
        """
        Инициализация VSL контроллера (статический режим для обратной совместимости).

        Args:
            traci_conn: Активное соединение TraCI.
            idm_v0_default: Исходное ограничение скорости (v0) на участках VSL (м/с). 
                            Используется как верхний предел для v_VSL и для восстановления.
            ctrl_segments_lanes: Список ID полос ('lane_id'), которые будут контролироваться VSL.
            ts_control_interval: Интервал времени (в секундах) между шагами управления VSL.
            kp, ki, kd: Коэффициенты ПИД-регулятора.
            v_min_vsl_limit: Минимальное значение скорости, устанавливаемое VSL (м/с).
            rho_crit_target: Целевая критическая плотность (ТС/км), которую должен поддерживать контроллер.
            vsl_detector_id: ID детектора E1 (induction loop) в SUMO для сбора данных.
            vsl_detector_length_m: Длина зоны детектирования этого детектора (м).
            log_csv_full_path: Полный путь к файлу для сохранения логов VSL.
                               Если пустая строка или None, логирование отключается.
            sim_step_length: Длительность одного шага симуляции SUMO (с). Используется для
                             определения количества шагов симуляции в одном контрольном интервале Ts.
            enabled: Флаг, включен ли контроллер при инициализации.
        """
        self.traci_conn = traci_conn
        self.idm_v0_default = idm_v0_default
        self.ctrl_segments_lanes = ctrl_segments_lanes if ctrl_segments_lanes else []
        self.Ts = ts_control_interval
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.v_min = v_min_vsl_limit
        self.rho_crit = rho_crit_target
        self.vsl_detector_id = vsl_detector_id
        self.camera_dx = vsl_detector_length_m 
        self.log_csv_path = log_csv_full_path
        self.sim_step_length = sim_step_length

        self.enabled = enabled
        self.initial_max_speeds: Dict[str, float] = {}

        self.u_k_minus_1 = 0.0
        self.e_k_minus_1 = 0.0
        self.e_k_minus_2 = 0.0
        
        self.current_vsl_speed_m_s = self.idm_v0_default

        # Параметры для динамического управления (устанавливаются в create_with_scenario)
        self.vsl_optimizer = None
        self.scenario = None 
        self.vsl_params = None
        self.v_max_vsl_limit = self.idm_v0_default
        self.use_dynamic_control = False

        if self.sim_step_length <= 1e-6: # Близко к нулю или отрицательное
            logger.warning(f"sim_step_length ({self.sim_step_length}с) некорректен. "
                           f"VSL может работать неправильно. Установлен в 1 симуляционный шаг на контрольный интервал.")
            self.control_steps_to_wait = 1
        elif self.Ts < self.sim_step_length:
            logger.warning(f"Интервал управления VSL Ts ({self.Ts}с) меньше шага симуляции ({self.sim_step_length}с). "
                           f"VSL будет срабатывать на каждом шаге симуляции.")
            self.control_steps_to_wait = 1
        else:
            self.control_steps_to_wait = max(1, int(round(self.Ts / self.sim_step_length)))
        
        self.steps_since_last_control = self.control_steps_to_wait # Чтобы первый вызов step() привел к срабатыванию, если Ts позволяет

        self._store_and_apply_initial_speeds()
        self._setup_logging()

        logger.info(f"VSLController инициализирован. Детектор: {self.vsl_detector_id}, Зона VSL: {self.ctrl_segments_lanes}")
        logger.info(f"Параметры VSL: Ts={self.Ts:.2f}с, Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, v_min={self.v_min:.2f}м/с, rho_crit={self.rho_crit:.2f}тс/км")
        logger.info(f"Начальная скорость VSL установлена на {self.idm_v0_default:.2f} м/с.")
        logger.info(f"VSL будет срабатывать каждые {self.control_steps_to_wait} шагов симуляции "
                    f"(Ts={self.Ts:.2f}с, sim_step={self.sim_step_length:.2f}с).")

    def _store_and_apply_initial_speeds(self):
        """Сохраняет исходные максимальные скорости на управляемых участках и применяет начальную скорость VSL."""
        if not self.ctrl_segments_lanes:
            logger.warning("Нет указанных сегментов для VSL (ctrl_segments_lanes пуст). VSL не будет управлять скоростями.")
            return

        try:
            for lane_id in self.ctrl_segments_lanes:
                if lane_id not in self.traci_conn.lane.getIDList():
                    logger.warning(f"Полоса {lane_id} для VSL не найдена в симуляции. Пропускается.")
                    continue
                self.initial_max_speeds[lane_id] = self.traci_conn.lane.getMaxSpeed(lane_id)
                self.traci_conn.lane.setMaxSpeed(lane_id, self.idm_v0_default)
            logger.info(f"Начальные скорости сохранены. VSL активна, скорость установлена на {self.idm_v0_default} м/с для {len(self.ctrl_segments_lanes)} полос.")
        except TraCIException as e:
            logger.error(f"Ошибка TraCI при сохранении/установке начальных скоростей: {e}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при сохранении/установке начальных скоростей: {e}")

    def _setup_logging(self):
        """Настраивает CSV-логгер."""
        self.log_file = None
        self.csv_writer = None
        if self.log_csv_path:
            try:
                log_dir = os.path.dirname(self.log_csv_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                self.log_file = open(self.log_csv_path, 'w', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.log_file)
                self.csv_writer.writerow([
                    "sim_time_s", "enabled", "num_vehicles_detector", "avg_speed_detector_m_s",
                    "flow_veh_per_h", "density_veh_per_km", "error_rho",
                    "pid_delta_u_k", "pid_u_k", "vsl_target_speed_m_s", "vsl_applied_speed_m_s"
                ])
                logger.info(f"Логирование VSL будет сохранено в: {self.log_csv_path}")
            except IOError as e:
                logger.error(f"Не удалось открыть файл лога VSL {self.log_csv_path}: {e}")
                self.csv_writer = None
        else:
            logger.info("Путь к файлу лога VSL не указан, логирование отключено.")

    def step(self, sim_time: float):
        """
        Выполняет один шаг логики VSL-контроллера.
        Этот метод должен вызываться на каждом шаге симуляции SUMO.
        Логика управления VSL срабатывает только каждые self.Ts секунд.
        
        Поддерживает два режима:
        1. Статический PID контроль (стандартный)
        2. Динамическое управление с математически обоснованными сценариями
        """
        self.steps_since_last_control += 1
        
        # Логирование даже если отключено или не время для шага, но с частотой Ts
        is_control_time_event = self.steps_since_last_control >= self.control_steps_to_wait

        if not self.enabled:
            if is_control_time_event:
                if self.csv_writer:
                    try:
                        self.csv_writer.writerow([
                            f"{sim_time:.2f}", int(self.enabled), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                            self.current_vsl_speed_m_s, self.current_vsl_speed_m_s
                        ])
                    except Exception as e:
                        logger.warning(f"Ошибка записи в лог VSL (когда отключен): {e}")
                self.steps_since_last_control = 0 
            return

        if not is_control_time_event:
            return 

        self.steps_since_last_control = 0 

        num_vehicles_last_step = 0
        avg_speed_m_s = 0.0
        flow_veh_per_h = 0.0
        density_veh_per_km = 0.0
        error_rho = 0.0
        delta_u_k = 0.0
        u_k = self.u_k_minus_1 
        vsl_target_speed = self.current_vsl_speed_m_s

        try:
            if self.vsl_detector_id in self.traci_conn.inductionloop.getIDList():
                num_vehicles_last_step = self.traci_conn.inductionloop.getLastStepVehicleNumber(self.vsl_detector_id)
                flow_veh_per_h = (num_vehicles_last_step / self.sim_step_length) * 3600 if self.sim_step_length > 0 else 0
                
                speed_last_step_m_s = self.traci_conn.inductionloop.getLastStepMeanSpeed(self.vsl_detector_id)
                if speed_last_step_m_s < 0: 
                    speed_last_step_m_s = self.current_vsl_speed_m_s 
                avg_speed_m_s = speed_last_step_m_s
            else:
                logger.warning(f"Детектор {self.vsl_detector_id} не найден. Данные для VSL не получены.")
                # Логируем как есть и выходим из управляющей логики
                if self.csv_writer:
                    self.csv_writer.writerow([
                        f"{sim_time:.2f}", int(self.enabled), num_vehicles_last_step, f"{avg_speed_m_s:.2f}",
                        f"{flow_veh_per_h:.2f}", f"{density_veh_per_km:.2f}", f"{error_rho:.2f}",
                        f"{delta_u_k:.3f}", f"{u_k:.3f}", f"{vsl_target_speed:.2f}", f"{self.current_vsl_speed_m_s:.2f}"
                    ])
                return

            # Расчет плотности
            if avg_speed_m_s > 0.1:
                density_veh_per_km = flow_veh_per_h / (avg_speed_m_s * 3.6)
            else:
                density_veh_per_km = (1000.0 / 7.0) if num_vehicles_last_step > 0 else 0.0
            
            # НОВОЕ: Выбор режима управления
            if self.use_dynamic_control and self.vsl_optimizer:
                # Динамическое управление с математически обоснованными формулами
                vsl_target_speed = self._calculate_dynamic_vsl(density_veh_per_km, avg_speed_m_s)
                
                # Для совместимости с логированием, рассчитываем PID-подобные значения
                error_rho = density_veh_per_km - self.rho_crit
                delta_u_k = self.idm_v0_default - vsl_target_speed
                u_k = delta_u_k
                
                # Обновляем состояние для непрерывности
                self.e_k_minus_2 = self.e_k_minus_1
                self.e_k_minus_1 = error_rho
                self.u_k_minus_1 = u_k
            else:
                # Стандартное PID управление
                error_rho = density_veh_per_km - self.rho_crit

                delta_u_k = (self.Kp * (error_rho - self.e_k_minus_1) +
                             self.Ki * self.Ts * error_rho)
                if self.Ts > 1e-6:
                     delta_u_k += self.Kd * (error_rho - 2 * self.e_k_minus_1 + self.e_k_minus_2) / self.Ts
                
                u_k = self.u_k_minus_1 + delta_u_k
                
                vsl_target_speed = self.idm_v0_default - u_k 
                
                # Обновляем состояние PID
                self.e_k_minus_2 = self.e_k_minus_1
                self.e_k_minus_1 = error_rho
                self.u_k_minus_1 = u_k

            # Применяем ограничения
            v_min_limit = self.v_min
            v_max_limit = self.v_max_vsl_limit if self.use_dynamic_control else self.idm_v0_default
            vsl_target_speed = max(v_min_limit, min(vsl_target_speed, v_max_limit))

            # Применяем VSL к полосам движения
            self.current_vsl_speed_m_s = vsl_target_speed
            for lane_id in self.ctrl_segments_lanes:
                if lane_id in self.traci_conn.lane.getIDList():
                    self.traci_conn.lane.setMaxSpeed(lane_id, self.current_vsl_speed_m_s)

        except TraCIException as e:
            logger.error(f"Ошибка TraCI в VSLController.step(): {e}")
        except Exception as e:
            logger.exception(f"Неожиданная ошибка в VSLController.step(): {e}")

        # Логирование результатов
        if self.csv_writer:
            try:
                self.csv_writer.writerow([
                    f"{sim_time:.2f}", int(self.enabled), num_vehicles_last_step, f"{avg_speed_m_s:.2f}",
                    f"{flow_veh_per_h:.2f}", f"{density_veh_per_km:.2f}", f"{error_rho:.2f}",
                    f"{delta_u_k:.3f}", f"{u_k:.3f}", f"{vsl_target_speed:.2f}", f"{self.current_vsl_speed_m_s:.2f}"
                ])
            except Exception as e:
                logger.warning(f"Ошибка записи в лог VSL: {e}")
    
    def _calculate_dynamic_vsl(self, current_density: float, current_speed: float) -> float:
        """
        Расчет VSL с использованием математически обоснованных формул.
        
        Args:
            current_density: Текущая плотность (авто/км)
            current_speed: Текущая средняя скорость (м/с)
            
        Returns:
            Целевая скорость VSL (м/с)
        """
        if not self.vsl_optimizer or not self.scenario:
            logger.warning("VSL оптимизатор или сценарий не настроены. Используется базовая скорость.")
            return self.idm_v0_default
        
        try:
            # Рассчитываем управляющее воздействие для текущего сценария
            vsl_speed = self.vsl_optimizer.calculate_vsl_control(
                scenario=self.scenario,
                current_density=current_density
            )
            
            logger.debug(f"Динамическое VSL ({self.scenario}): ρ={current_density:.1f} -> VSL={vsl_speed:.1f} м/с")
            
            return vsl_speed
            
        except Exception as e:
            logger.error(f"Ошибка при расчете динамического VSL: {e}")
            return self.current_vsl_speed_m_s  # Возвращаем текущее значение как fallback

    def toggle(self, enabled: bool, sim_time: Optional[float] = None):
        """Включает или отключает VSL контроллер."""
        if self.enabled == enabled:
            return 

        self.enabled = enabled
        current_time_str = f" (sim_time: {sim_time:.2f}s)" if sim_time is not None else ""

        if not self.enabled:
            logger.info(f"VSL контроллер ОТКЛЮЧЕН{current_time_str}. Восстановление исходных скоростей...")
            try:
                for lane_id, original_speed in self.initial_max_speeds.items():
                    if lane_id in self.traci_conn.lane.getIDList():
                        self.traci_conn.lane.setMaxSpeed(lane_id, original_speed)
                self.current_vsl_speed_m_s = self.idm_v0_default 
                self.u_k_minus_1 = 0.0
                self.e_k_minus_1 = 0.0
                self.e_k_minus_2 = 0.0
                logger.info("Исходные скорости восстановлены.")
            except TraCIException as e:
                logger.error(f"Ошибка TraCI при восстановлении скоростей: {e}")
            except Exception as e:
                logger.error(f"Неожиданная ошибка при восстановлении скоростей: {e}")
        else: 
            logger.info(f"VSL контроллер ВКЛЮЧЕН{current_time_str}.")
            # Применяем текущую (которая была сброшена на idm_v0_default при выключении или осталась с инициализации)
            # или последнюю рассчитанную скорость VSL, если контроллер не выключался и это первый вызов toggle(True)
            # Так как current_vsl_speed_m_s сбрасывается на idm_v0_default при выключении, 
            # то при включении мы начнем с idm_v0_default.
            for lane_id in self.ctrl_segments_lanes:
                 if lane_id in self.traci_conn.lane.getIDList():
                    self.traci_conn.lane.setMaxSpeed(lane_id, self.current_vsl_speed_m_s) 
            logger.info(f"VSL скорость {self.current_vsl_speed_m_s:.2f} м/с применена к {len(self.ctrl_segments_lanes)} полосам.")

    def close_log(self):
        """Закрывает файл лога, если он был открыт."""
        if self.log_file:
            try:
                self.log_file.close()
                logger.info(f"Файл лога VSL {self.log_csv_path} закрыт.")
            except Exception as e:
                logger.error(f"Ошибка при закрытии файла лога VSL: {e}")
        self.log_file = None
        self.csv_writer = None

# --- Вспомогательные функции ---

def compute_idm_partials(v_e: float, s_e: float, 
                         idm_v0: float, idm_T: float, idm_a: float, idm_delta: float
                         ) -> Tuple[float, float, float]:
    """
    Вычисляет парциальные производные f_s, f_v, f_v0 для IDM.
    v_e: равновесная скорость (м/с)
    s_e: равновесное расстояние headway (м) = 1/плотность (1/rho_e)
    idm_v0, idm_T, idm_a, idm_delta: параметры IDM
    """
    if idm_v0 <= 1e-6 or s_e <= 1e-6: # Избегаем деления на ноль и некорректных значений
        # logger.debug(f"compute_idm_partials: Некорректные входные данные v0={idm_v0}, s_e={s_e}. Возврат нулей.")
        return 0.0, 0.0, 0.0

    # v_e не может быть больше v0 для стандартного IDM в равновесии, но может быть в переходных процессах.
    # Однако, если v_e > v0, (v_e/v0)^delta может быть > 1, что делает (1 - v_ratio_delta) отрицательным.
    # Это может привести к отрицательной f_s, что соответствует нестабильности (фантомные пробки).
    # Ограничим v_e сверху значением idm_v0 для расчета производных в контексте стабильного потока.
    v_e_capped = min(v_e, idm_v0)

    v_ratio_delta = (v_e_capped / idm_v0) ** idm_delta
    
    # f_s = ∂(dv/dt)/∂s
    # (2 * a / s_e) * (s_star/s_e)^2 * (s_star_deriv_s) - это из другой формулы
    # Используем формулу из теории в начале файла: 2*a/s_e * (1 - (v_e/v0)^delta)
    f_s = (2 * idm_a / s_e) * (1 - v_ratio_delta)

    # f_v = ∂(dv/dt)/∂v
    # -a * [ delta * (v_e/v0)^(delta-1) / v0 + 2*T/s_e * (1 - (v_e/v0)^delta) ]
    # Нужно аккуратно с (v_e/v0)^(delta-1) если v_e=0 и delta-1 < 0
    if v_e_capped <= 1e-6 and idm_delta < 1.0: # например, delta = 0.5, v_e_capped=0 -> 0^(-0.5) -> inf
        term1_fv = 0.0 # В этой точке производная по скорости может быть не определена или очень велика
    else:
        term1_fv = idm_delta * (v_e_capped / idm_v0)**(idm_delta - 1.0) / idm_v0
        
    term2_fv = (2 * idm_T / s_e) * (1 - v_ratio_delta)
    f_v = -idm_a * (term1_fv + term2_fv)

    # f_v0 = ∂(dv/dt)/∂v0
    # a * delta * (v_e/v0)^delta / v0
    f_v0 = idm_a * idm_delta * v_ratio_delta / idm_v0
    
    # logger.debug(f"IDM Partials: v_e={v_e_capped:.2f}, s_e={s_e:.2f}, v0={idm_v0:.2f}, T={idm_T:.2f}, a={idm_a:.2f}, delta={idm_delta:.1f}")
    # logger.debug(f"  -> f_s={f_s:.4f}, f_v={f_v:.4f}, f_v0={f_v0:.4f}")
    return f_s, f_v, f_v0

# Основной блок if __name__ == "__main__": удален, т.к. это модуль