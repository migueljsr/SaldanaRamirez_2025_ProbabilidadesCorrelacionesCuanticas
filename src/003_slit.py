"""
NOMBRE: 003_slit.py
DESCRIPCIÓN: Simulación No-Trivial de Doble Rendija mediante Dinámica de Ondas.
             VERSIÓN OPTIMIZADA: Usa Batching para GPU de 4GB VRAM.
AUTOR: Miguel J. Saldaña - Asistencia (Gemini 3.0 - GTP 5.1)
FECHA: 21 Nov 2025
HARDWARE: GPU Acceleration (CUDA/CuPy)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# ==========================================
# 1. CONFIGURACIÓN DEL ENTORNO Y RUTAS
# ==========================================
SCRIPT_NAME = "003_slit"
BASE_OUTPUT_DIR = "./data"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, SCRIPT_NAME)

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[SISTEMA] Directorio de resultados: {OUTPUT_DIR}")
except Exception as e:
    print(f"[ERROR] No se pudo crear el directorio: {e}")
    sys.exit(1)

# ==========================================
# 2. GESTIÓN DE HARDWARE (GPU/CPU)
# ==========================================
try:
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    USE_GPU = True
    print("[HARDWARE] Modo: Aceleración Vectorizada Sim (CuPy)")
except ImportError:
    import numpy as cp

    USE_GPU = False
    print("[HARDWARE] Modo: CPU (NumPy) - Lento.")

# ==========================================
# 3. PARÁMETROS FÍSICOS Y TÉCNICOS
# ==========================================
LAMBDA = 1.0
K_WAVE = 2 * np.pi / LAMBDA
SLIT_DIST = 40.0 * LAMBDA
SCREEN_DIST = 2000.0 * LAMBDA
SCREEN_WIDTH = 400.0 * LAMBDA

# --- AJUSTE CRÍTICO DE MEMORIA ---
N_SHOTS = 100_000  # Total de eventos estadísticos (Alta calidad)
BATCH_SIZE = 5_000  # Procesamos de 5,000 en 5,000 para no saturar 4GB VRAM
N_PIXELS = 4096
SIGMA_STEPS = 30


# ==========================================
# 4. MOTOR FÍSICO OPTIMIZADO
# ==========================================
class SlitSimulator:
    def __init__(self, n_pixels, screen_width, slit_dist, screen_dist, k_wave):
        self.n_pixels = n_pixels
        self.d = slit_dist
        self.L = screen_dist
        self.k = k_wave
        self.width = screen_width
        self._init_geometry()

    def _init_geometry(self):
        x_span = cp.linspace(-self.width / 2, self.width / 2, self.n_pixels, dtype=cp.float32)
        self.r1 = cp.sqrt(self.L ** 2 + (x_span - self.d / 2) ** 2)
        self.r2 = cp.sqrt(self.L ** 2 + (x_span + self.d / 2) ** 2)
        # Fasores base (reutilizables)
        self.psi1_geo = cp.exp(1j * self.k * self.r1).astype(cp.complex64)
        self.psi2_geo = cp.exp(1j * self.k * self.r2).astype(cp.complex64)
        self.x_axis = x_span

    def run_ensemble(self, sigma_phase, n_shots):
        """
        Ejecuta la simulación por lotes para ahorrar memoria VRAM.
        """
        # Acumulador de intensidad (float32) inicia en 0 en la GPU
        I_accum = cp.zeros(self.n_pixels, dtype=cp.float32)

        # Procesar por lotes
        processed = 0
        while processed < n_shots:
            # Determinar tamaño del lote actual
            current_batch_size = min(BATCH_SIZE, n_shots - processed)

            # 1. Ruido de fase (Axioma 3)
            phi1 = cp.random.normal(0, sigma_phase, (current_batch_size, 1), dtype=cp.float32)
            phi2 = cp.random.normal(0, sigma_phase, (current_batch_size, 1), dtype=cp.float32)

            # 2. Construcción de campos (Broadcasting: Batch x Pixels)
            # Esto consume ~160MB por matriz con Batch=5000
            field_total = (self.psi1_geo[None, :] * cp.exp(1j * phi1)) + \
                          (self.psi2_geo[None, :] * cp.exp(1j * phi2))

            # 3. Resonancia instantánea (Intensidad)
            I_batch = cp.abs(field_total) ** 2

            # 4. Sumar al acumulador total
            # Sumamos axis=0 (colapsamos el lote) y añadimos al acumulador global
            I_accum += cp.sum(I_batch, axis=0)

            # Limpieza de memoria inmediata
            del phi1, phi2, field_total, I_batch
            if USE_GPU:
                mempool.free_all_blocks()

            processed += current_batch_size

        # Promedio final
        I_mean = I_accum / n_shots
        return I_mean


# ==========================================
# 5. FUNCIONES AUXILIARES
# ==========================================
def calculate_visibility_robust(I_gpu):
    if USE_GPU:
        I = cp.asnumpy(I_gpu)
    else:
        I = I_gpu

    if np.max(I) == 0: return 0.0
    I = I / np.max(I)

    center_idx = len(I) // 2
    window = len(I) // 10
    I_center = I[center_idx - window: center_idx + window]

    val_max = np.percentile(I_center, 99.9)
    val_min = np.percentile(I_center, 0.1)

    if val_max + val_min == 0: return 0.0
    V = (val_max - val_min) / (val_max + val_min)
    return np.clip(V, 0.0, 1.0)


def theoretical_curve(sigma):
    return np.exp(-sigma ** 2)


# ==========================================
# 6. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("-" * 60)
    print(f"INICIANDO SIMULACIÓN: {SCRIPT_NAME} (OPTIMIZADO 4GB VRAM)")
    print("-" * 60)

    sim = SlitSimulator(N_PIXELS, SCREEN_WIDTH, SLIT_DIST, SCREEN_DIST, K_WAVE)
    sigmas = np.linspace(0.05, 3.0, SIGMA_STEPS)
    visibilities = []

    # Almacenar perfiles
    profile_coherent = None
    profile_incoherent = None

    t_start = time.time()

    for i, s in enumerate(sigmas):
        # Ejecutar con manejo de memoria
        I_mean = sim.run_ensemble(s, N_SHOTS)
        v = calculate_visibility_robust(I_mean)
        visibilities.append(v)

        # Guardar extremos
        if i == 0:
            profile_coherent = cp.asnumpy(I_mean) / float(cp.max(I_mean))
        if i == SIGMA_STEPS - 1:
            profile_incoherent = cp.asnumpy(I_mean) / float(cp.max(I_mean))

        print(f" > Paso {i + 1}/{SIGMA_STEPS}: Sigma={s:.2f} -> V={v:.4f}")

    print(f"[FIN] Tiempo total: {time.time() - t_start:.2f} s")

    # Guardar CSV
    df = pd.DataFrame({
        "sigma": sigmas,
        "vis_sim": visibilities,
        "vis_teo": theoretical_curve(sigmas)
    })
    csv_path = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_data.csv")
    df.to_csv(csv_path, index=False)

    # Graficar Perfiles
    x = cp.asnumpy(sim.x_axis) / LAMBDA
    plt.figure(figsize=(10, 6))
    plt.plot(x, profile_coherent, 'b', label=f'Coherente (V={visibilities[0]:.2f})')
    plt.plot(x, profile_incoherent, 'r--', label=f'Clásico (V={visibilities[-1]:.2f})')
    plt.title("Sim: Emergencia de la Realidad Clásica")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_profiles.png"), dpi=300)
    plt.close()

    # Graficar Decoherencia
    r2 = r2_score(visibilities, theoretical_curve(sigmas))
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, visibilities, 'bo', label='Simulación')
    plt.plot(sigmas, theoretical_curve(sigmas), 'k--', label='Teoría')
    plt.title(f"Validación Decoherencia (R2={r2:.4f})")
    plt.xlabel("Sigma (Ruido de Fase)")
    plt.ylabel("Visibilidad")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_decoherence.png"), dpi=300)
    plt.close()

    print("=== FINALIZADO CORRECTAMENTE ===")