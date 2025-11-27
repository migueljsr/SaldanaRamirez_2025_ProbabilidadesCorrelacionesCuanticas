"""
NOMBRE: 008_genesis.py (Versión Cosmológica/Geométrica)
DESCRIPCIÓN: Emergencia de la Cuantización.
             SOLUCIÓN ONTOLÓGICA: La 'Sopa Primordial' no es uniforme.
             Sigue la distribución del índice de refracción n_eff.
             Las ondas se concentran donde el potencial es bajo y se diluyen
             donde es alto, evitando inestabilidades numéricas de borde.
AUTOR: Miguel J. Saldaña - Asistencia (Gemini 3.0 - GTP 5.1)
HARDWARE: GPU Acceleration (CUDA/CuPy)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
SCRIPT_NAME = "008_genesis"
BASE_OUTPUT_DIR = "./data"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, SCRIPT_NAME)

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[SISTEMA] Directorio: {OUTPUT_DIR}")
except Exception as e:
    sys.exit(1)

try:
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    USE_GPU = True
    print("[HARDWARE] GPU CuPy Activada.")
except ImportError:
    import numpy as cp

    USE_GPU = False
    print("[HARDWARE] CPU NumPy.")

# ==========================================
# 2. PARÁMETROS FÍSICOS Y NUMÉRICOS (CALIBRADOS V3)
# ==========================================
MASS = 1.0
HBAR = 1.0
OMEGA = 1.0

# Parámetros Numéricos
# L=16.0 reduce V_max en los bordes a 32.0.
# DT=0.01 garantiza fase < 0.32 rads (Muy estable).
L_BOX = 16.0
N_POINTS = 1024
DT = 0.01
T_MAX = 1000.0         # Mantenemos tiempo largo para precisión espectral

# Ensamble
N_ENSEMBLE = 50        # 50 universos son suficientes si la física es correcta

# ==========================================
# 3. MOTOR DE EVOLUCIÓN
# ==========================================
class TOPEnsembleGenesis:
    def __init__(self, n_points, l_box, mass, hbar, dt, n_ensemble):
        self.N = n_points
        self.L = l_box
        self.m = mass
        self.hbar = hbar
        self.dt = dt
        self.dx = self.L / self.N
        self.n_ens = n_ensemble

        self.x = cp.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False, dtype=cp.float32)
        self.k = 2 * cp.pi * cp.fft.fftfreq(self.N, d=self.dx).astype(cp.float32)

        T_op = (self.hbar ** 2 * self.k ** 2) / (2 * self.m)
        self.U_kinetic = cp.exp(-1j * T_op * self.dt / self.hbar).astype(cp.complex64)
        self.U_potential_half = None
        self.V_array = None  # Guardamos V para usarlo en la inicialización

    def set_potential(self):
        # V(x) = 0.5 * m * w^2 * x^2
        self.V_array = 0.5 * self.m * (OMEGA ** 2) * (self.x ** 2)
        self.U_potential_half = cp.exp(-1j * self.V_array * (self.dt / 2) / self.hbar).astype(cp.complex64)

    def initialize_ensemble_soup(self):
        """
        Estado Inicial: SOPA GEOMÉTRICA (TOP Paper 1).

        En lugar de cortar artificialmente, usamos la física del índice de refracción.
        Las ondas primordiales no pueden tener amplitud alta en regiones de
        "índice bajo" (Potencial alto).

        Aplicamos un peso estadístico de Boltzmann local: A ~ exp(-V(x)/E_scale)
        Esto simula que el vacío se "acomoda" a la geometría del espacio-tiempo.
        """
        # 1. Generar Ruido Blanco Puro (Caos de fase)
        real = cp.random.normal(0, 1, (self.n_ens, self.N), dtype=cp.float32)
        imag = cp.random.normal(0, 1, (self.n_ens, self.N), dtype=cp.float32)
        psi = real + 1j * imag

        # 2. PONDERACIÓN GEOMÉTRICA (La Clave de TOP)
        # Definimos una "Temperatura de Vacío" o escala de energía E_scale.
        # Debe ser suficiente para poblar los primeros niveles, pero decaer en V alto.
        # E_scale ~ 10.0 cubre los primeros niveles n=0..9
        E_scale = 10.0

        # El "Índice de Densidad de Ondas"
        # Si V(x) > E_scale, la amplitud cae exponencialmente.
        density_weight = cp.exp(-self.V_array / E_scale)

        # Aplicar peso al ruido (Broadcasting)
        psi *= density_weight[None, :]

        # 3. Normalizar
        norms = cp.sqrt(cp.sum(cp.abs(psi) ** 2, axis=1, keepdims=True) * self.dx)
        psi /= norms

        return psi.astype(cp.complex64)

    def step(self, psi):
        psi *= self.U_potential_half[None, :]
        psi_k = cp.fft.fft(psi, axis=1)
        psi_k *= self.U_kinetic[None, :]
        psi = cp.fft.ifft(psi_k, axis=1)
        psi *= self.U_potential_half[None, :]
        return psi


# ==========================================
# 4. EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    print("-" * 60)
    print(f"INICIANDO GÉNESIS (MODO COSMOLÓGICO): {SCRIPT_NAME}")
    print("Justificación: La densidad de ondas sigue la geometría del potencial (Paper 1).")
    print("-" * 60)

    sim = TOPEnsembleGenesis(N_POINTS, L_BOX, MASS, HBAR, DT, N_ENSEMBLE)
    sim.set_potential()

    # Estado Inicial "Geométrico"
    psi_0 = sim.initialize_ensemble_soup()
    psi = psi_0.copy()
    psi_0_conj = cp.conj(psi_0)

    autocorr_avg = []
    n_steps = int(T_MAX / DT)

    print(f"[INFO] Evolucionando {N_ENSEMBLE} universos ({n_steps} pasos)...")
    t_start = time.time()

    for step in range(n_steps):
        overlaps = cp.sum(psi_0_conj * psi, axis=1) * sim.dx
        avg_overlap = cp.mean(overlaps)
        autocorr_avg.append(complex(avg_overlap))

        psi = sim.step(psi)

        if step % 5000 == 0:
            print(f"   > {step / n_steps * 100:.0f}% ...")

    print(f"[INFO] Tiempo: {time.time() - t_start:.2f} s")

    # ==========================================
    # 5. ANÁLISIS
    # ==========================================
    signal = np.array(autocorr_avg)
    window = np.blackman(len(signal))
    signal_w = signal * window

    fft_vals = np.fft.fft(signal_w, n=len(signal) * 4)
    freqs = np.fft.fftfreq(len(fft_vals), d=DT)

    energies = HBAR * 2 * np.pi * freqs
    power = np.abs(fft_vals) ** 2

    mask = (energies > 0.1) & (energies < 8.0)
    E_plot = energies[mask]
    P_plot = power[mask]
    P_plot /= np.max(P_plot)

    peaks_idx, _ = find_peaks(P_plot, height=0.05, distance=30)
    E_found = E_plot[peaks_idx]

    print("-" * 65)
    print(f"Nivel (n) | E_sim (TOP)      | E_teo (QM)       | Error %")
    print("-" * 65)

    for i, e_sim in enumerate(E_found):
        if i > 6: break
        e_teo = 0.5 + i
        err = abs(e_sim - e_teo) / e_teo * 100
        print(f"{i:9d} | {e_sim:.4f}           | {e_teo:.4f}           | {err:.2f}%")

    # Guardar
    df = pd.DataFrame({"Energy": E_plot, "Power": P_plot})
    df.to_csv(os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_spectrum.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(E_plot, P_plot, 'b-', label='Espectro Emergente (Sopa Geométrica)')
    plt.plot(E_found, P_plot[peaks_idx], 'ro')
    for n in range(7): plt.axvline(0.5 + n, color='k', alpha=0.2, linestyle='--')
    plt.title("Génesis de Niveles: Resonancia de Vacío Ponderado por Potencial")
    plt.xlabel("Energía")
    plt.xlim(0, 7.5)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_spectrum.png"), dpi=300)
    plt.close()

    print("\n=== VALIDACIÓN FINALIZADA ===")