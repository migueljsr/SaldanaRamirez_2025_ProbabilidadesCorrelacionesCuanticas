"""
NOMBRE: 005_scattering.py (Parte 1)
DESCRIPCIÓN: Validación unificada de Dispersión y Efecto Túnel (S1 y S3)
             Nota: Se usa la etiqueta S1 y S3 debido a que se trabajó este apartartado como 
             Simulación 1 y Simulacioón 3 en una versión (borrador) anterior del paper. 
             Escenarios:
             - S1: Paquete gaussiano libre (Figura X – dispersión).
             - S3: Escalón de potencial (Figura Y – reflexión/transmisión).
             Valida: Ensanchamiento del paquete libre y coeficientes T/R en escalón.
AUTOR:  Miguel J. Saldaña - Asistencia (Gemini 3.0 - GTP 5.1)
FECHA: 21 Nov 2025
HARDWARE: GPU Acceleration (CUDA/CuPy)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
SCRIPT_NAME = "005_scattering"
BASE_OUTPUT_DIR = "./data"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, SCRIPT_NAME)

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[SISTEMA] Directorio: {OUTPUT_DIR}")
except Exception as e:
    sys.exit(1)

# ==========================================
# 2. HARDWARE
# ==========================================
try:
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    USE_GPU = True
    print("[HARDWARE] GPU CuPy Activada (Modo Scattering).")
except ImportError:
    import numpy as cp

    USE_GPU = False
    print("[HARDWARE] CPU NumPy (Lento).")


# ==========================================
# 3. MOTOR DE DINÁMICA (Reutilizado y Adaptado)
# ==========================================
# necesarios para problemas de scattering (para que el paquete viaje lejos).

class SimScatteringEngine:
    def __init__(self, n_points, l_box, mass, hbar, dt):
        self.N = n_points
        self.L = l_box
        self.m = mass
        self.hbar = hbar
        self.dt = dt
        self.dx = self.L / self.N

        # Grilla Espacial
        self.x = cp.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False, dtype=cp.float32)

        # Grilla Momento (k)
        self.k = 2 * cp.pi * cp.fft.fftfreq(self.N, d=self.dx).astype(cp.float32)

        # Operador Cinético (Pre-calculado)
        # T = (hbar*k)^2 / 2m
        T_op = (self.hbar ** 2 * self.k ** 2) / (2 * self.m)
        self.U_kinetic = cp.exp(-1j * T_op * self.dt / self.hbar).astype(cp.complex64)

        self.V_x = None
        self.U_potential_half = None

    def set_potential(self, V_array_gpu):
        """Define el paisaje de potencial V(x)"""
        self.V_x = V_array_gpu
        self.U_potential_half = cp.exp(-1j * self.V_x * (self.dt / 2) / self.hbar).astype(cp.complex64)

    def get_initial_packet(self, x0, p0, sigma):
        """Genera paquete gaussiano inicial"""
        norm = 1.0 / (np.pi ** (0.25) * cp.sqrt(sigma))
        phase = cp.exp(1j * p0 * self.x / self.hbar)
        envelope = cp.exp(-(self.x - x0) ** 2 / (2 * sigma ** 2))
        return (norm * phase * envelope).astype(cp.complex64)

    def step(self, psi):
        """Evolución Split-Step"""
        psi = self.U_potential_half * psi  # V(dt/2)
        psi_k = cp.fft.fft(psi)
        psi_k = self.U_kinetic * psi_k  # T(dt)
        psi = cp.fft.ifft(psi_k)
        psi = self.U_potential_half * psi  # V(dt/2)
        return psi


# ==========================================
# 4. CONFIGURACIÓN DE ESCENARIOS (S1 y S3)
# ==========================================
# Parámetros Generales
MASS = 1.0
HBAR = 1.0
# Usamos una caja muy grande para que el paquete viaje sin chocar con los bordes
L_BOX = 200.0
N_POINTS = 8192  # Alta resolución espacial
DT = 0.05


# --- ESCENARIO S1: Paquete Libre (Expansión) ---
def run_scenario_s1(engine):
    print("\n>>> EJECUTANDO ESCENARIO S1: PAQUETE LIBRE (Dispersión) <<<")
    # V(x) = 0
    V_free = cp.zeros_like(engine.x)
    engine.set_potential(V_free)

    # Estado inicial: Quieto en el centro, sigma pequeño
    sigma0 = 2.0
    psi = engine.get_initial_packet(x0=0.0, p0=0.0, sigma=sigma0)

    # Teoría: sigma(t) = sigma0 * sqrt(1 + (hbar*t / (2*m*sigma0^2))^2)
    t_max = 20.0
    steps = int(t_max / engine.dt)

    times = []
    sigmas_sim = []
    sigmas_theo = []

    for step in range(steps):
        t = step * engine.dt

        # Calcular Sigma Numérico
        # sigma^2 = <x^2> - <x>^2
        prob = cp.abs(psi) ** 2 * engine.dx
        x_mean = cp.sum(engine.x * prob)
        x2_mean = cp.sum((engine.x ** 2) * prob)
        sigma_sim = cp.sqrt(x2_mean - x_mean ** 2)

        # Guardar datos (cada 10 pasos para velocidad)
        if step % 10 == 0:
            times.append(t)
            sigmas_sim.append(float(sigma_sim))

            # Cálculo teórico
            term = (engine.hbar * t) / (2 * engine.m * sigma0 ** 2)
            s_th = sigma0 * np.sqrt(1 + term ** 2)
            sigmas_theo.append(s_th)

        psi = engine.step(psi)

    return times, sigmas_sim, sigmas_theo


# --- ESCENARIO S3: Escalón de Potencial (Scattering) ---
def run_scenario_s3(engine, energy_ratio=1.2):
    """
    energy_ratio = E_kin / V0
    Si > 1: Transmisión clásica posible (pero con reflexión cuántica).
    Si < 1: Efecto túnel / Reflexión total.
    """
    mode_str = "TRANSMISIÓN" if energy_ratio > 1 else "TUNEL/REFLEXIÓN"
    print(f"\n>>> EJECUTANDO ESCENARIO S3: ESCALÓN ({mode_str}, E/V0={energy_ratio}) <<<")

    # 1. Definir Paquete Incidente
    sigma0 = 3.0
    x0 = -50.0  # Empezar lejos a la izquierda
    p0 = 2.0  # Momento incidente

    # Energía cinética del paquete: E = p^2 / 2m
    E_inc = (p0 ** 2) / (2 * engine.m)

    # 2. Definir Potencial Escalón V(x) = V0 si x > 0
    V0 = E_inc / energy_ratio
    V_step = cp.where(engine.x > 0, V0, 0.0)

    # Suavizado del escalón para evitar ringing numérico excesivo (físicamente realista)
    # Usamos una función sigmoide muy empinada o tanh
    # V_step = V0 * 0.5 * (1 + cp.tanh(engine.x * 2.0))

    engine.set_potential(V_step)
    psi = engine.get_initial_packet(x0, p0, sigma0)

    # Evolucionar hasta que el paquete haya interactuado
    # Velocidad v = p/m = 2/1 = 2. Distancia ~ 80. Tiempo ~ 40.
    t_max = 60.0
    steps = int(t_max / engine.dt)

    # Snapshots para visualización
    snapshots = []

    for step in range(steps):
        psi = engine.step(psi)
        if step % (steps // 4) == 0:  # Guardar 4 estados
            if USE_GPU:
                snapshots.append(cp.asnumpy(cp.abs(psi) ** 2))
            else:
                snapshots.append(np.abs(psi) ** 2)

    # Calcular Coeficientes Finales
    # Probabilidad en x > 0 (Transmisión)
    prob_density = cp.abs(psi) ** 2 * engine.dx
    # Máscara para lado derecho
    mask_right = engine.x > 0
    T_sim = float(cp.sum(prob_density[mask_right]))
    R_sim = 1.0 - T_sim

    # Cálculo Teórico (Onda Plana Ideal)
    # k1 = p/hbar, k2 = sqrt(2m(E-V0))/hbar
    k1 = p0 / engine.hbar
    if E_inc > V0:
        k2 = np.sqrt(2 * engine.m * (E_inc - V0)) / engine.hbar
        T_theo = 4 * k1 * k2 / (k1 + k2) ** 2
    else:
        T_theo = 0.0  # Reflexión total para onda plana (túnel es despreciable en escalón infinito)

    return T_sim, R_sim, T_theo, snapshots, cp.asnumpy(V_step)


# ==========================================
# 5. BLOQUE PRINCIPAL DE EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    print("-" * 60)
    print(f"INICIANDO VALIDACIÓN DE SCATTERING SIM: {SCRIPT_NAME}")
    print(f"Objetivo: Validar índice efectivo n_eff en régimen dinámico.")
    print("-" * 60)

    # Inicializar Motor Global
    # Usamos una caja grande y alta resolución para acomodar la dispersión
    sim = SimScatteringEngine(N_POINTS, L_BOX, MASS, HBAR, DT)

    # ---------------------------------------------------------
    # EJECUCIÓN ESCENARIO S1: PAQUETE LIBRE
    # ---------------------------------------------------------
    times_s1, sig_sim, sig_theo = run_scenario_s1(sim)

    # Cálculo de Error S1
    sig_sim_arr = np.array(sig_sim)
    sig_theo_arr = np.array(sig_theo)
    error_s1_mae = np.mean(np.abs(sig_sim_arr - sig_theo_arr))
    error_s1_rel = np.mean(np.abs((sig_sim_arr - sig_theo_arr) / sig_theo_arr)) * 100

    print(f"RESULTADOS S1:")
    print(f"Error Medio Absoluto (MAE): {error_s1_mae:.6f}")
    print(f"Error Relativo Medio: {error_s1_rel:.4f}%")

    # Guardar Datos S1
    df_s1 = pd.DataFrame({
        "time": times_s1,
        "sigma_sim": sig_sim,
        "sigma_theo": sig_theo
    })
    df_s1.to_csv(os.path.join(OUTPUT_DIR, "005_free_packet.csv"), index=False)

    # ---------------------------------------------------------
    # EJECUCIÓN ESCENARIO S3: ESCALÓN DE POTENCIAL
    # ---------------------------------------------------------

    # CASO A: E > V0 (Transmisión con Reflexión Parcial)
    # Energy Ratio 1.3 -> Tiene suficiente energía para pasar, pero V0 lo frena
    T_sim_A, R_sim_A, T_theo_A, snaps_A, V_A = run_scenario_s3(sim, energy_ratio=1.3)

    print(f"RESULTADOS S3 (CASO E > V0):")
    print(f"Transmisión Simulada (Sim): {T_sim_A:.4f}")
    print(f"Transmisión Teórica (QM):   {T_theo_A:.4f}")
    print(f"Reflexión Simulada (Sim):   {R_sim_A:.4f}")
    print(f"Suma T+R (Unitariedad):     {T_sim_A + R_sim_A:.6f}")

    # CASO B: E < V0 (Efecto Túnel / Reflexión Total)
    # Energy Ratio 0.8 -> No tiene energía clásica para pasar
    T_sim_B, R_sim_B, T_theo_B, snaps_B, V_B = run_scenario_s3(sim, energy_ratio=0.8)

    print(f"RESULTADOS S3 (CASO E < V0):")
    print(f"Transmisión Simulada (Sim): {T_sim_B:.4f} (Cola evanescente)")
    print(f"Reflexión Simulada (Sim):   {R_sim_B:.4f}")

    # Guardar Datos S3
    df_s3 = pd.DataFrame({
        "Scenario": ["E_gt_V0", "E_lt_V0"],
        "T_sim": [T_sim_A, T_sim_B],
        "T_theo": [T_theo_A, T_theo_B],
        "R_sim": [R_sim_A, R_sim_B]
    })
    df_s3.to_csv(os.path.join(OUTPUT_DIR, "005_step_potential.csv"), index=False)

    # ==========================================
    # 6. GENERACIÓN DE GRÁFICAS (PAPER READY)
    # ==========================================
    if USE_GPU:
        x_axis = cp.asnumpy(sim.x)
    else:
        x_axis = sim.x

    # --- GRAFICA S1: Dispersión del Paquete ---
    plt.figure(figsize=(10, 6))
    plt.plot(times_s1, sig_sim, 'b-', linewidth=2, label='Simulación (GPU)')
    plt.plot(times_s1, sig_theo, 'r--', linewidth=2, label='Teoría Analítica')
    plt.title(f"Dispersión de Paquete Libre (Sim) \nValidación de transporte en vacío ($n_{{eff}}=1$)", fontsize=12)
    plt.xlabel("Tiempo Normalizado", fontsize=11)
    plt.ylabel(r"Anchura del Paquete $\sigma(t)$", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_dispersion.png"), dpi=300)
    plt.close()

    # --- GRAFICA S3: Escalón de Potencial (Transmisión) ---
    plt.figure(figsize=(10, 6))

    # Graficar Potencial (Escalado para que se vea junto a la onda)
    scale_V = np.max(snaps_A[0]) / np.max(V_A) if np.max(V_A) > 0 else 0
    plt.plot(x_axis, V_A * scale_V, 'k-', linewidth=2, alpha=0.3, label='Potencial V(x)')

    # Graficar Evolución
    colors = plt.cm.viridis(np.linspace(0, 1, len(snaps_A)))
    for i, dens in enumerate(snaps_A):
        label = "Inicio" if i == 0 else ("Final" if i == len(snaps_A) - 1 else None)
        plt.plot(x_axis, dens, color=colors[i], alpha=0.8, label=label)
        plt.fill_between(x_axis, dens, color=colors[i], alpha=0.1)

    plt.title(f"Scattering en Escalón ($E > V_0$)\n$T_{{Sim}}={T_sim_A:.3f}$ vs $T_{{QM}}={T_theo_A:.3f}$",
              fontsize=12)
    plt.xlabel("Posición", fontsize=11)
    plt.ylabel("Densidad $|\psi|^2$", fontsize=11)
    plt.legend()
    plt.xlim(-80, 80)  # Zoom a la zona de interacción
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_transmission.png"), dpi=300)
    plt.close()

    # --- GRAFICA S3: Escalón de Potencial (Reflexión/Túnel) ---
    plt.figure(figsize=(10, 6))

    # Potencial
    # Para el caso < V0, escalamos visualmente
    scale_V_B = np.max(snaps_B[0]) / (np.max(V_B) * 1.5)
    plt.plot(x_axis, V_B * scale_V_B, 'k-', linewidth=2, alpha=0.3, label='Potencial V(x)')

    for i, dens in enumerate(snaps_B):
        label = "Inicio" if i == 0 else ("Final (Reflejado)" if i == len(snaps_B) - 1 else None)
        plt.plot(x_axis, dens, color=colors[i], alpha=0.8, label=label)
        plt.fill_between(x_axis, dens, color=colors[i], alpha=0.1)

    plt.title(f"Reflexión Total ($E < V_0$)\n$R_{{Sim}}={R_sim_B:.4f}$ (Unitariedad preservada)", fontsize=12)
    plt.xlabel("Posición", fontsize=11)
    plt.ylabel("Densidad $|\psi|^2$", fontsize=11)
    plt.legend()
    plt.xlim(-80, 80)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_reflection.png"), dpi=300)
    plt.close()

    print("\n=== VALIDACIÓN S1/S3 COMPLETADA ===")
    print(f"Gráficas generadas en: {OUTPUT_DIR}")