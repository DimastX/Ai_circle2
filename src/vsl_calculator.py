#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vsl_calculator.py

Модуль для расчета оптимальных параметров Variable Speed Limit (VSL) управления
на основе математически обоснованных сценариев.

Реализует четыре сценария оптимизации:
1. speed - оптимизация средней скорости
2. variance - минимизация дисперсии скорости  
3. wave - подавление stop-and-go волн
4. throughput - максимизация пропускной способности

Все параметры рассчитываются динамически на основе времени реакции водителя T' 
и анализа устойчивости IDM.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class VSLParams:
    """Параметры VSL контроллера для конкретного сценария."""
    kp: float  # Пропорциональный коэффициент PID
    ki: float  # Интегральный коэффициент PID
    kd: float  # Дифференциальный коэффициент PID
    target_density: float  # Целевая плотность (автомобилей/км)
    vsl_bounds: Tuple[float, float]  # Границы VSL (мин, макс) в м/с
    scenario_name: str  # Название сценария
    
    def __post_init__(self):
        """Валидация параметров после инициализации."""
        if self.kp < 0:
            logger.warning(f"Отрицательный Kp={self.kp} может привести к нестабильности")
        if self.ki < 0:
            logger.warning(f"Отрицательный Ki={self.ki} может привести к нестабильности")
        if self.kd < 0:
            logger.warning(f"Отрицательный Kd={self.kd} может привести к нестабильности")
        if self.target_density <= 0:
            raise ValueError(f"target_density должна быть положительной: {self.target_density}")
        if self.vsl_bounds[0] > self.vsl_bounds[1]:
            raise ValueError(f"Некорректные границы VSL: {self.vsl_bounds}")


class VSLOptimizer:
    """
    Класс для расчета оптимальных параметров VSL управления.
    
    Основан на математических формулах для четырех сценариев оптимизации,
    учитывающих время реакции водителя T' и анализ устойчивости IDM.
    """
    
    def __init__(self, T_prime: float, idm_params: Optional[dict] = None):
        """
        Инициализация VSL оптимизатора.
        
        Args:
            T_prime: Время реакции водителя (с), диапазон 0.5-1.5
            idm_params: Параметры IDM модели (опционально)
        """
        if not (0.5 <= T_prime <= 1.5):
            logger.warning(f"T_prime={T_prime} вне рекомендуемого диапазона 0.5-1.5с")
        
        self.T_prime = T_prime
        
        # Базовые параметры IDM
        if idm_params is None:
            idm_params = {}
        
        self.s0 = idm_params.get('s0', 2.0)      # Минимальная дистанция (м)
        self.v0 = idm_params.get('v0', 30.0)    # Желаемая скорость (м/с)
        self.T = idm_params.get('T', 1.5)       # Время следования (с)
        self.a = idm_params.get('a', 1.0)       # Максимальное ускорение (м/с²)
        self.b = idm_params.get('b', 1.5)       # Комфортное торможение (м/с²)
        self.delta = idm_params.get('delta', 4)  # Экспонента скорости
        self.L_veh = idm_params.get('L_veh', 5.0)  # Длина автомобиля (м)
        
        # Рассчитываем базовую критическую плотность
        self.rho_crit_base = self.calculate_critical_density()
        
        logger.info(f"VSLOptimizer инициализирован: T'={T_prime:.2f}с, ρ_crit_base={self.rho_crit_base:.2f}")
    
    def calculate_critical_density(self) -> float:
        """
        Расчет критической плотности на основе фундаментальной диаграммы IDM.
        
        Формула: ρ_crit = 1000/(s_0 + v_0*T + L_veh) * (1 - 0.3*(T'/0.9)²)
        
        Returns:
            Критическая плотность в автомобилях/км
        """
        s_star = self.s0 + self.v0 * self.T
        rho_base = 1000.0 / (s_star + self.L_veh)
        
        # Адаптация под время реакции водителя
        T_prime_normalized = self.T_prime / 0.9
        adaptation_factor = 1.0 - 0.3 * (T_prime_normalized ** 2)
        
        rho_crit = rho_base * adaptation_factor
        
        if rho_crit <= 0:
            logger.error(f"Критическая плотность ≤ 0: {rho_crit}")
            rho_crit = 1.0  # Минимальное значение для избежания ошибок
        
        return rho_crit
    
    def calculate_stability_margin(self, f_s: float, f_v: float, f_dv: float) -> float:
        """
        Расчет запаса устойчивости для сценария минимизации дисперсии.
        
        Формула: stability_margin = exp(-T'/0.9) * (f_v²/2 - f_dv*f_v - f_s)
        
        Args:
            f_s: Частная производная по расстоянию
            f_v: Частная производная по скорости
            f_dv: Частная производная по относительной скорости
            
        Returns:
            Запас устойчивости
        """
        exp_factor = math.exp(-self.T_prime / 0.9)
        stability_expression = (f_v ** 2) / 2.0 - f_dv * f_v - f_s
        return exp_factor * stability_expression
    
    def calculate_vsl_control(self, scenario: str, current_density: float, 
                            stability_data: Optional[dict] = None) -> float:
        """
        Расчет управляющего воздействия VSL для заданного сценария.
        
        Args:
            scenario: Название сценария ('speed', 'variance', 'wave', 'throughput')
            current_density: Текущая плотность (автомобилей/км)
            stability_data: Данные анализа устойчивости (для сценария 'variance')
            
        Returns:
            Значение VSL в м/с
        """
        rho = current_density
        rho_crit = self.get_adaptive_critical_density(scenario)
        
        if scenario == "speed":
            # VSL(t) = v_max * (1 - (ρ(t)/ρ_crit(T'))^0.7)
            v_max = self.v0
            if rho_crit > 0:
                ratio = min(rho / rho_crit, 1.0)  # Ограничиваем отношение
                vsl = v_max * (1.0 - (ratio ** 0.7))
            else:
                vsl = v_max
                
        elif scenario == "variance":
            # VSL(t) = 100 * (1 - ρ(t)/(0.8 * ρ_crit(T')))
            if rho_crit > 0:
                ratio = rho / (0.8 * rho_crit)
                vsl = 100.0 * max(0.0, 1.0 - ratio)  # Ограничиваем снизу
            else:
                vsl = 100.0
                
        elif scenario == "wave":
            # VSL(t) = 80 * (1 + 0.5 * tanh(5 * (ρ_crit(T') - ρ(t))))
            if rho_crit > 0:
                arg = 5.0 * (rho_crit - rho)
                tanh_val = math.tanh(arg)
                vsl = 80.0 * (1.0 + 0.5 * tanh_val)
            else:
                vsl = 80.0
                
        elif scenario == "throughput":
            # VSL(t) = q_max/ρ(t) * (1 - exp(-ρ(t)/ρ_crit(T')))
            q_max = self.calculate_q_max(rho_crit)
            if rho > 0 and rho_crit > 0:
                exp_term = math.exp(-rho / rho_crit)
                vsl = (q_max / rho) * (1.0 - exp_term)
            else:
                vsl = self.v0  # Безопасное значение по умолчанию
        
        elif scenario == "none":
            # Baseline: без VSL управления
            kp = 0.0  # Отключаем PID
            ki = 0.0
            kd = 0.0
            target_density = rho_crit  # Используем базовую критическую плотность
            vsl_bounds = (self.v0, self.v0)  # Фиксированная скорость
            
        else:
            raise ValueError(f"Неизвестный сценарий: {scenario}")
        
        # Применяем разумные ограничения
        vsl = max(5.0, min(vsl, self.v0))  # Ограничиваем диапазон [5, v0]
        
        return vsl
    
    def get_adaptive_critical_density(self, scenario: str) -> float:
        """
        Получить адаптивную критическую плотность для сценария.
        
        Args:
            scenario: Название сценария
            
        Returns:
            Адаптивная критическая плотность
        """
        if scenario == "wave":
            # Адаптивная критическая плотность для подавления волн
            if self.T_prime > 0.9:
                return 0.6 * self.rho_crit_base
            else:
                return 0.75 * self.rho_crit_base
        else:
            return self.rho_crit_base
    
    def calculate_q_max(self, rho_crit: float) -> float:
        """
        Расчет максимального потока для сценария пропускной способности.
        
        Формула: q_max = ρ_crit(T') * v_opt * (1 - ρ_crit(T')/ρ_jam)
        
        Args:
            rho_crit: Критическая плотность
            
        Returns:
            Максимальный поток (автомобилей/с)
        """
        # Оптимальная скорость при критической плотности
        v_opt = self.v0 * 0.8  # Примерно 80% от желаемой скорости
        
        # Плотность затора (приблизительная оценка)
        rho_jam = 1000.0 / self.L_veh  # Максимальная плотность
        
        if rho_jam > 0:
            q_max = rho_crit * v_opt * (1.0 - rho_crit / rho_jam)
        else:
            q_max = rho_crit * v_opt
        
        return max(0.0, q_max)  # Ограничиваем снизу
    
    def get_scenario_params(self, scenario: str, 
                          stability_data: Optional[dict] = None) -> VSLParams:
        """
        Получить параметры VSL для заданного сценария.
        
        Args:
            scenario: Название сценария ('speed', 'variance', 'wave', 'throughput')
            stability_data: Данные анализа устойчивости (опционально)
            
        Returns:
            VSLParams с коэффициентами для данного сценария
        """
        rho_crit = self.get_adaptive_critical_density(scenario)
        
        if scenario == "speed":
            # Сценарий 1: Оптимизация средней скорости
            kp = 0.6 + 0.2 * (self.T_prime / 0.9)
            ki = 0.12 * math.exp(-self.T_prime / 0.5)
            kd = 0.08 * (1.0 + self.T_prime / 0.9)
            target_density = 0.7 * rho_crit
            vsl_bounds = (15.0, 33.3)  # ~60-120 км/ч
            
        elif scenario == "variance":
            # Сценарий 2: Минимизация дисперсии скорости
            stability_margin = 1.0  # По умолчанию
            if stability_data:
                f_s = stability_data.get('f_s', 0.0)
                f_v = stability_data.get('f_v', 0.0)
                f_dv = stability_data.get('f_dv', 0.0)
                stability_margin = self.calculate_stability_margin(f_s, f_v, f_dv)
            
            stability_margin = min(stability_margin, 1.0)  # Ограничиваем сверху
            kp = 0.8 + 0.3 * stability_margin
            ki = 0.05 / (1.0 + 2.0 * self.T_prime)
            kd = 0.15 * (1.0 + 0.5 * self.T_prime)
            target_density = 0.8 * rho_crit
            vsl_bounds = (10.0, 30.0)  # Более консервативные границы
            
        elif scenario == "wave":
            # Сценарий 3: Подавление stop-and-go волн
            kp = 1.0 * (1.0 + 0.5 * self.T_prime / 0.9)
            ki = 0.08 * math.exp(-2.0 * self.T_prime / 0.9)
            kd = 0.2 * (1.0 + self.T_prime / 0.9)
            target_density = rho_crit  # Используем адаптивную критическую плотность
            vsl_bounds = (8.0, 25.0)  # Широкий диапазон для подавления волн
            
        elif scenario == "throughput":
            # Сценарий 4: Максимизация пропускной способности
            kp = 0.5 / (1.0 + self.T_prime)
            ki = 0.15 * (1.0 - 0.3 * self.T_prime)
            kd = 0.05  # Константа
            target_density = rho_crit * 0.9  # Близко к критической, но с запасом
            vsl_bounds = (12.0, 35.0)  # Оптимизированы для потока
        
        elif scenario == "none":
            # Baseline: без VSL управления
            kp = 0.0  # Отключаем PID
            ki = 0.0
            kd = 0.0
            target_density = rho_crit  # Используем базовую критическую плотность
            vsl_bounds = (self.v0, self.v0)  # Фиксированная скорость
        
        else:
            raise ValueError(f"Неизвестный сценарий: {scenario}")
        
        # Валидация коэффициентов
        kp = max(0.01, kp)  # Минимальные положительные значения
        ki = max(0.001, ki)
        kd = max(0.001, kd)
        
        return VSLParams(
            kp=kp,
            ki=ki, 
            kd=kd,
            target_density=target_density,
            vsl_bounds=vsl_bounds,
            scenario_name=scenario
        )
    
    def get_all_scenarios(self) -> dict:
        """
        Получить параметры для всех доступных сценариев.
        
        Returns:
            Словарь {scenario_name: VSLParams}
        """
        scenarios = {}
        for scenario in ['none', 'speed', 'variance', 'wave', 'throughput']:
            try:
                scenarios[scenario] = self.get_scenario_params(scenario)
            except Exception as e:
                logger.error(f"Ошибка при расчете параметров для сценария {scenario}: {e}")
        
        return scenarios
    
    def validate_scenario(self, scenario: str) -> bool:
        """
        Проверить корректность сценария.
        
        Args:
            scenario: Название сценария
            
        Returns:
            True если сценарий поддерживается
        """
        return scenario in ['none', 'speed', 'variance', 'wave', 'throughput']
    
    def get_scenario_description(self, scenario: str) -> str:
        """
        Получить описание сценария.
        
        Args:
            scenario: Название сценария
            
        Returns:
            Текстовое описание сценария
        """
        descriptions = {
            'none': 'Baseline без VSL управления - отсутствие активного управления скоростью',
            'speed': 'Оптимизация средней скорости - максимизация J = (1/T)∫v(t)dt',
            'variance': 'Минимизация дисперсии скорости - минимизация J = ∫(v(t)-v̄)²dt',
            'wave': 'Подавление stop-and-go волн - минимизация J = ∫(dv/dt)²dt',
            'throughput': 'Максимизация пропускной способности - максимизация J = ∫q(t)dt'
        }
        return descriptions.get(scenario, f"Неизвестный сценарий: {scenario}")


# Вспомогательные функции для интеграции с существующим кодом

def create_vsl_optimizer_from_args(args) -> VSLOptimizer:
    """
    Создать VSLOptimizer из аргументов командной строки.
    
    Args:
        args: Объект argparse с параметрами
        
    Returns:
        Настроенный VSLOptimizer
    """
    T_prime = getattr(args, 'T_prime', 0.9)
    
    # Извлекаем IDM параметры если они есть
    idm_params = {}
    if hasattr(args, 'idm_params') and args.idm_params:
        try:
            # Ожидаем формат: "v0,T,s0" например "30.0,1.5,2.0"
            params = args.idm_params.split(',')
            if len(params) >= 3:
                idm_params['v0'] = float(params[0])
                idm_params['T'] = float(params[1])
                idm_params['s0'] = float(params[2])
        except (ValueError, IndexError) as e:
            logger.warning(f"Не удалось распарсить idm_params: {e}")
    
    return VSLOptimizer(T_prime, idm_params)


def get_vsl_params_for_scenario(scenario: str, T_prime: float = 0.9, 
                               idm_params: Optional[dict] = None,
                               stability_data: Optional[dict] = None) -> VSLParams:
    """
    Удобная функция для получения VSL параметров.
    
    Args:
        scenario: Название сценария
        T_prime: Время реакции водителя
        idm_params: Параметры IDM (опционально)
        stability_data: Данные анализа устойчивости (опционально)
        
    Returns:
        VSLParams для заданного сценария
    """
    optimizer = VSLOptimizer(T_prime, idm_params)
    return optimizer.get_scenario_params(scenario, stability_data)


if __name__ == "__main__":
    # Тестирование модуля
    logging.basicConfig(level=logging.INFO)
    
    print("Тестирование VSLOptimizer...")
    
    # Тест с различными временами реакции
    for T_prime in [0.5, 0.9, 1.2]:
        print(f"\n=== T' = {T_prime}с ===")
        optimizer = VSLOptimizer(T_prime)
        
        for scenario in ['speed', 'variance', 'wave', 'throughput']:
            params = optimizer.get_scenario_params(scenario)
            print(f"{scenario:>10}: Kp={params.kp:.3f}, Ki={params.ki:.3f}, Kd={params.kd:.3f}, "
                  f"ρ_target={params.target_density:.1f}")
    
    print("\nТестирование управляющих воздействий...")
    optimizer = VSLOptimizer(0.9)
    
    # Тест управляющих воздействий для различных плотностей
    densities = [10, 20, 30, 40, 50]
    for scenario in ['speed', 'variance', 'wave', 'throughput']:
        print(f"\n{scenario.upper()} scenario:")
        for rho in densities:
            vsl = optimizer.calculate_vsl_control(scenario, rho)
            print(f"  ρ={rho:2d} -> VSL={vsl:.1f} м/с") 