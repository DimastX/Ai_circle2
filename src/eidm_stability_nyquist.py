
"""
E‑IDM Stability Toolkit (with Nyquist plot)
------------------------------------------
Author : ChatGPT (o3), 2025‑05‑07
Language: Python 3.x
Requires: numpy, scipy, matplotlib

Features
--------
* Решение равновесия при фиксированном потоке q
* Расчёт коэффициентов линейной модели (A,B,C)
* Графики
  1) Dispersion Re(λ) vs k
  2) Bode magnitude |G(jω)|
  3) Bode phase φ(ω)
  4) Nyquist годограф G(jω)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# ---------- математические функции ----------

def equilibrium_velocities(q, params, v_min=0.1, v_max=None, N=6000):
    """Find all positive roots v_e of equilibrium equation for given flow q (veh/s)."""
    v0, delta, s0, T, L = params['v0'], params['delta'], params['s0'], params['T'], params['L']
    if v_max is None:
        v_max = v0 * 1.2

    def f(v):
        s = v / q - L
        if s <= 0:
            return 1.0
        return (v / v0) ** delta + ((s0 + T * v) / s) ** 2 - 1.0

    x = np.linspace(v_min, v_max, N)
    fvals = np.array([f(v) for v in x])
    roots = []
    for i in range(N - 1):
        if np.sign(fvals[i]) * np.sign(fvals[i + 1]) < 0:
            try:
                root = brentq(f, x[i], x[i + 1])
                if all(abs(root - r) > 1e-3 for r in roots):
                    roots.append(root)
            except ValueError:
                pass
    return sorted(roots, reverse=True)  # descending speed order


def build_coefficients(v_e, params, q):
    """Return A,B,C,k2,s_e for given equilibrium v_e."""
    a = params['a']
    b_dec = params['b']
    v0, delta, s0, T, L = params['v0'], params['delta'], params['s0'], params['T'], params['L']
    s_e = v_e / q - L
    k2 = 1 - (v_e / v0) ** delta
    A = 2 * a * k2 / s_e
    B = a * (delta * (v_e / v0) ** (delta - 1) / v0 + 2 * T * k2 / s_e)
    C = a * v_e * k2 / (np.sqrt(a * b_dec) * s_e)
    return A, B, C, k2, s_e


def lambdas(k, A, B, C):
    """Eigenvalues λ1,λ2 for given k."""
    z = np.exp(-1j * k)
    b = B + C * (1 - z)
    discr = b ** 2 + 4 * A * (1 - z)
    sqrt_d = np.sqrt(discr)
    return 0.5 * (b + sqrt_d), 0.5 * (b - sqrt_d)


def transfer_function(s, k, A, B, C):
    """Return G(s) for given Laplace variable s and wave‑number k."""
    z = np.exp(-1j * k)
    return 1.0 / (s ** 2 - s * (B + C * (1 - z)) - A * (z - 1))


# ---------- графические функции ----------

def plot_dispersion(k, re1, re2, title):
    plt.figure()
    plt.plot(k, re1, label='λ1')
    plt.plot(k, re2, label='λ2')
    plt.axhline(0, linestyle='--')
    plt.xlabel('k [rad]')
    plt.ylabel('Re(λ) [1/s]')
    plt.title(title)
    plt.grid(True)
    plt.legend()


def plot_bode(omega, mag, phase):
    plt.figure()
    plt.semilogx(omega, 20 * np.log10(mag))
    plt.xlabel('ω [rad/s]')
    plt.ylabel('|G(jω)| [dB]')
    plt.title('Bode magnitude')
    plt.grid(True)

    plt.figure()
    plt.semilogx(omega, phase * 180 / np.pi)
    plt.xlabel('ω [rad/s]')
    plt.ylabel('Phase [deg]')
    plt.title('Bode phase')
    plt.grid(True)


def plot_nyquist(G_vals, title):
    plt.figure()
    plt.plot(G_vals.real, G_vals.imag)
    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel('Re(G(jω))')
    plt.ylabel('Im(G(jω))')
    plt.title(title)
    plt.grid(True)


# ---------- основной исполняемый код ----------

def analyse_one_root(v_e, params, q, root_idx=1):
    """Full set of plots for one equilibrium root."""
    A, B, C, k2, s_e = build_coefficients(v_e, params, q)
    k = np.linspace(1e-3, np.pi, 800)
    re1, re2 = [], []
    for ki in k:
        l1, l2 = lambdas(ki, A, B, C)
        re1.append(l1.real)
        re2.append(l2.real)
    plot_dispersion(k, re1, re2,
                    f'Корень #{root_idx}: v_e={v_e:.2f} м/с, s_e={s_e:.2f} м')

    # Найдём критический k (максимум Reλ)
    re_max = np.maximum(re1, re2)
    idx_max = int(np.argmax(re_max))
    k_crit = k[idx_max]

    # Bode
    omega = np.logspace(-2, 1.8, 500)
    G = transfer_function(1j * omega[:, None], k_crit, A, B, C)
    mag = np.abs(G[:, 0])
    phase = np.angle(G[:, 0])
    plot_bode(omega, mag, phase)

    # Nyquist — положительные + зеркально отрицательные частоты
    G_pos = G[:, 0]
    G_full = np.concatenate([G_pos, np.conjugate(G_pos[::-1])])
    plot_nyquist(G_full, f'Nyquist (k={k_crit:.3f})')

    # вывод суб‑резюме
    print(f"Root #{root_idx}: v_e={v_e:.3f} m/s, s_e={s_e:.3f} m, "
          f"k_crit={k_crit:.3f}, Reλ_max={re_max[idx_max]:.3f}")


def main(params):
    """Run analysis for given params."""
    q_h = params['q']
    q = q_h / 3600.0
    roots = equilibrium_velocities(q, params)
    if not roots:
        print("Нет допустимых равновесий при данном q")
        return
    print("Found roots:", [f"{v:.2f}" for v in roots])
    for idx, v_e in enumerate(roots, 1):
        analyse_one_root(v_e, params, q, idx)
    plt.show()


if __name__ == "__main__":
    params = {
        "a": 1.2,     # m/s^2
        "b": 2.0,     # m/s^2 decel
        "v0": 30.0,   # desired speed m/s
        "delta": 4,
        "s0": 2.0,    # min gap m
        "T": 1.3,     # headway s
        "L": 5.0,     # vehicle length m
        "q": 1100,    # veh/h
    }
    main(params)
