import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd  # Добавлен импорт pandas для анализа симуляционных данных
import os

# ----------- Параметры EIDM (синхронизированы с run_straight.py) ------------------
a = 2.6
b = 4.5
s0 = 1.0  # min_gap_mean из run_straight.py
v0 = 20.0
delta = 4.0
tau = 0.5  # tau_mean из run_straight.py
q_fixed = 0.769  # vehicles/m/s

# ----------- Helper functions -------------
def coefficients_from_qT(a, b, s0, T, v0, delta, q):
    if q * T >= 1:
        return np.nan, np.nan, np.nan
    v_e = q * s0 / (1 - q * T)
    s_e = s0 / (1 - q * T)
    term1 = (v_e / v0) ** delta
    A = -a * (delta / v_e * term1 + 2 * T * (s0 + T * v_e) / s_e ** 2)
    B = a * v_e * (s0 + T * v_e) / (np.sqrt(a * b) * s_e ** 2)
    C = 2 * a * (s0 + T * v_e) ** 2 / s_e ** 3
    return A, B, C

def omega_val(D, C):
    omega2 = 0.5 * (D ** 2 + np.sqrt(D ** 4 + 16 * C ** 2))
    return np.sqrt(omega2)

# ----------- 2D Graph: Re(lambda) vs T при tau=0.5 ------------
T_vals = np.linspace(0.1, 5.0, 500)
tau_fixed = 0.5
Re_lambda_vs_T = []
for T in T_vals:
    A, B, C = coefficients_from_qT(a, b, s0, T, v0, delta, q_fixed)
    if np.isnan(A) or np.isnan(B) or np.isnan(C):
        Re_lambda_vs_T.append(np.nan)
        continue
    D = A + 2 * B
    omega = omega_val(D, C)
    phi = omega * tau_fixed
    Re_lam = D * np.cos(phi) - omega * np.sin(phi)
    Re_lambda_vs_T.append(Re_lam)

# --- Анализ симуляционных данных и добавление экспериментальной точки ---
sim_dir = 'data/simulations/straight_results_N100_q0.769_accel2.6_decel4.5_20250503_221335'
sim_csv = os.path.join(sim_dir, 'density_flow_data.csv')
sim_csv_traj = os.path.join(sim_dir, 'simulation_data.csv')

if os.path.exists(sim_csv) and os.path.exists(sim_csv_traj):
    df = pd.read_csv(sim_csv)
    q_exp = df['flow'].mean()
    v_exp = df['mean_speed'].mean()
    rho_exp = df['density'].mean()
    if np.isnan(rho_exp):
        print("Плотность: не удалось вычислить (NaN)")
    else:
        print(f"Плотность (из density_flow_data.csv): {rho_exp:.4f}")
    s0_exp = s0
    a_exp = a
    b_exp = b
    v0_exp = v0
    delta_exp = delta
    T_exp = (1 - rho_exp * s0_exp) / q_exp
    print(f"T_exp: {T_exp:.4f}")
    print(f"q_exp: {q_exp:.4f}")
    print(f"rho_exp: {rho_exp:.4f}")
    print(f"s0_exp: {s0_exp:.4f}")
    v_e_exp = q_exp / rho_exp
    s_e_exp = 1 / rho_exp
    term1 = (v_e_exp / v0_exp) ** delta_exp
    A_exp = -a_exp * (delta_exp / v_e_exp * term1 + 2 * T_exp * (s0_exp + T_exp * v_e_exp) / s_e_exp ** 2)
    B_exp = a_exp * v_e_exp * (s0_exp + T_exp * v_e_exp) / (np.sqrt(a_exp * b_exp) * s_e_exp ** 2)
    C_exp = 2 * a_exp * (s0_exp + T_exp * v_e_exp) ** 2 / s_e_exp ** 3
    D_exp = A_exp + 2 * B_exp
    omega_exp = omega_val(D_exp, C_exp)
    phi_exp = omega_exp * tau_fixed
    Re_lambda_exp_tau = D_exp * np.cos(phi_exp) - omega_exp * np.sin(phi_exp)
    fig1, ax1 = plt.subplots()
    ax1.plot(T_vals, Re_lambda_vs_T, label=f"tau={tau_fixed} c, q={q_fixed} (аналитика)")
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.set_xlabel("Time gap T [s]")
    ax1.set_ylabel("Re lambda")
    ax1.set_title(f"Устойчивость потока: Re(lambda)(T) при tau={tau_fixed}")
    ax1.legend()
    ax1.grid(True)
    ax1.scatter([T_exp], [Re_lambda_exp_tau], color='red', label='Симуляция (SUMO)', zorder=5)
    ax1.annotate(f"T={T_exp:.2f}\nReλ={Re_lambda_exp_tau:.2f}",
                (T_exp, Re_lambda_exp_tau),
                textcoords="offset points", xytext=(10,10), ha='left', color='red')
    plt.show()

    tau_vals = np.linspace(0.1, 5.0, 500)
    Re_lambda_vs_tau = []
    for tau_var in tau_vals:
        omega_exp_tau = omega_val(D_exp, C_exp)
        phi_exp_tau = omega_exp_tau * tau_var
        Re_lambda_tau = D_exp * np.cos(phi_exp_tau) - omega_exp_tau * np.sin(phi_exp_tau)
        Re_lambda_vs_tau.append(Re_lambda_tau)
    fig2, ax2 = plt.subplots()
    ax2.plot(tau_vals, Re_lambda_vs_tau, label=f"T={T_exp:.2f} c, q={q_fixed} (аналитика)")
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_xlabel("tau [s]")
    ax2.set_ylabel("Re lambda")
    ax2.set_title(f"Устойчивость потока: Re(lambda)(tau) при T={T_exp:.2f}")
    ax2.legend()
    ax2.grid(True)
    ax2.scatter([tau_fixed], [Re_lambda_exp_tau], color='red', label='Симуляция (SUMO)', zorder=5)
    ax2.annotate(f"tau={tau_fixed:.2f}\nReλ={Re_lambda_exp_tau:.2f}",
                (tau_fixed, Re_lambda_exp_tau),
                textcoords="offset points", xytext=(10,10), ha='left', color='red')
    plt.show()
else:
    fig1, ax1 = plt.subplots()
    ax1.plot(T_vals, Re_lambda_vs_T, label=f"tau={tau_fixed} c, q={q_fixed} (аналитика)")
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.set_xlabel("Time gap T [s]")
    ax1.set_ylabel("Re lambda")
    ax1.set_title(f"Устойчивость потока: Re(lambda)(T) при tau={tau_fixed}")
    ax1.legend()
    ax1.grid(True)
    plt.show()

    tau_vals = np.linspace(0.1, 5.0, 500)
    Re_lambda_vs_tau = []
    T_fixed = 2.0
    A, B, C = coefficients_from_qT(a, b, s0, T_fixed, v0, delta, q_fixed)
    if not (np.isnan(A) or np.isnan(B) or np.isnan(C)):
        D = A + 2 * B
        for tau_var in tau_vals:
            omega_tau = omega_val(D, C)
            phi_tau = omega_tau * tau_var
            Re_lambda_tau = D * np.cos(phi_tau) - omega_tau * np.sin(phi_tau)
            Re_lambda_vs_tau.append(Re_lambda_tau)
        fig2, ax2 = plt.subplots()
        ax2.plot(tau_vals, Re_lambda_vs_tau, label=f"T={T_fixed} c, q={q_fixed} (аналитика)")
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_xlabel("tau [s]")
        ax2.set_ylabel("Re lambda")
        ax2.set_title(f"Устойчивость потока: Re(lambda)(tau) при T={T_fixed}")
        ax2.legend()
        ax2.grid(True)
        plt.show()

# ----------- 3D Graph: Re(lambda) vs (T, tau) -----
T_grid, tau_grid = np.meshgrid(np.linspace(0.5, 3.0, 100),
                               np.linspace(0.1, 4.0, 100))
Re_lambda_3D = np.full_like(T_grid, np.nan)

for i in range(T_grid.shape[0]):
    for j in range(T_grid.shape[1]):
        T = T_grid[i, j]
        tau_ = tau_grid[i, j]
        A, B, C = coefficients_from_qT(a, b, s0, T, v0, delta, q_fixed)
        if np.isnan(A) or np.isnan(B) or np.isnan(C):
            continue
        D = A + 2 * B
        omega = omega_val(D, C)
        phi = omega * tau_
        Re_lam = D * np.cos(phi) - omega * np.sin(phi)
        Re_lambda_3D[i, j] = Re_lam

# Plot 3D
fig = plt.figure(figsize=(10, 7))
ax3d = fig.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(T_grid, tau_grid, Re_lambda_3D, cmap='viridis', edgecolor='none')
ax3d.set_xlabel(r"T [s]")
ax3d.set_ylabel(r"$\tau_r$ [s]")
ax3d.set_zlabel(r"$\mathrm{Re}\,\lambda$")
ax3d.set_title(r"$\mathrm{Re}\,\lambda(T,\tau_r)$ при постоянном потоке $q$")
fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
plt.show()
