"""
NOMBRE: 004_eraser.py
DESCRIPCIÓN: Simulación Ontológica del Borrador Cuántico (Quantum Eraser).
             Demuestra que la 'Información' es geometría en el espacio interno Lambda.
             Escenario A: Caminos Ortogonales (Marcado) -> Sin Interferencia.
             Escenario B: Proyección Geométrica (Borrado) -> Recuperación de Interferencia.
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

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
SCRIPT_NAME = "004_eraser"
BASE_OUTPUT_DIR = "./data"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, SCRIPT_NAME)

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[SISTEMA] Directorio: {OUTPUT_DIR}")
except Exception as e:
    sys.exit(1)
    
    

# ==========================================
# 2. HARDWARE (GPU)
# ==========================================
try:
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    USE_GPU = True
    print("[HARDWARE] GPU CuPy Activada (Modo Vectorial Ontológico).")
except ImportError:
    import numpy as cp

    USE_GPU = False
    print("[HARDWARE] CPU NumPy (Lento).")

# ==========================================
# 3. PARÁMETROS FÍSICOS
# ==========================================
LAMBDA = 1.0
K_WAVE = 2 * np.pi / LAMBDA
SLIT_DIST = 40.0 * LAMBDA
SCREEN_DIST = 2000.0 * LAMBDA
SCREEN_WIDTH = 400.0 * LAMBDA

# Parámetros de Simulación Estocástica
N_PIXELS = 4096
N_SHOTS = 100_000  # Estadística masiva para emergencia suave
BATCH_SIZE = 5_000  # Gestión de memoria VRAM

# Coherencia: Usamos alta coherencia de fase (Sigma bajo) para demostrar
# que la pérdida de interferencia es GEOMÉTRICA (información), no por desorden.
SIGMA_PHASE = 0.05


# ==========================================
# 4. MOTOR ONTOLÓGICO: VECTORES EN LAMBDA
# ==========================================
class SimEraserEngine:
    def __init__(self, n_pixels, screen_width, slit_dist, screen_dist, k_wave):
        self.n_pixels = n_pixels
        self.width = screen_width
        self.d = slit_dist
        self.L = screen_dist
        self.k = k_wave

        self._init_geometry()

    def _init_geometry(self):
        """Pre-calcula caminos ópticos"""
        x_span = cp.linspace(-self.width / 2, self.width / 2, self.n_pixels, dtype=cp.float32)
        self.r1 = cp.sqrt(self.L ** 2 + (x_span - self.d / 2) ** 2)
        self.r2 = cp.sqrt(self.L ** 2 + (x_span + self.d / 2) ** 2)

        # Fasores base (Amplitud escalar compleja)
        # Psi = exp(i * k * r)
        self.psi1_base = cp.exp(1j * self.k * self.r1).astype(cp.complex64)
        self.psi2_base = cp.exp(1j * self.k * self.r2).astype(cp.complex64)

        self.x_axis = x_span

    def run_experiment(self, eraser_angle_deg=None):
        """
        Ejecuta el experimento de ensamble.

        Args:
            eraser_angle_deg (float or None):
                - Si es None: Escenario MARCADO.
                  Rendija 1 tiene vector interno |V> (1,0)
                  Rendija 2 tiene vector interno |H> (0,1)
                  Son ortogonales en Lambda.

                - Si es float (ej. 45.0): Escenario BORRADO.
                  Se aplica un filtro proyectivo en el ángulo dado antes de detectar.
                  Esto fuerza a ambos vectores a colapsar en una base común.
        """
        if eraser_angle_deg is None:
            mode = "MARCADO (Ortogonal)"
            u_eraser = None
        else:
            mode = f"BORRADO (Ángulo {eraser_angle_deg}°)"
            theta = np.deg2rad(eraser_angle_deg)
            # Vector unitario del polarizador/borrador
            u_eraser = cp.array([np.cos(theta), np.sin(theta)], dtype=cp.float32)

        print(f"   > Simulando: {mode} ...")

        I_accum = cp.zeros(self.n_pixels, dtype=cp.float32)
        processed = 0

        # Vectores base ortogonales en el espacio interno
        # Rendija 1 -> Vertical (1, 0)
        # Rendija 2 -> Horizontal (0, 1)
        # Shape para broadcasting: (1, 1, 2) -> (Batch, Pixel, Componente XY)
        vec_1 = cp.array([1.0, 0.0], dtype=cp.float32).reshape(1, 1, 2)
        vec_2 = cp.array([0.0, 1.0], dtype=cp.float32).reshape(1, 1, 2)

        while processed < N_SHOTS:
            current_batch = min(BATCH_SIZE, N_SHOTS - processed)

            # 1. Ruido de fase (Hipótesis 3 - Fluctuaciones del vacío/fuente)
            # Aunque S es alto, existe ruido aleatorio
            phi1 = cp.random.normal(0, SIGMA_PHASE, (current_batch, 1), dtype=cp.float32)
            phi2 = cp.random.normal(0, SIGMA_PHASE, (current_batch, 1), dtype=cp.float32)

            # 2. Construcción de Campos Vectoriales (Broadcasting)
            # Scalar_Field * Phase_Noise * Internal_Vector
            # Dim: (Batch, Pixels, 1) * (1, 1, 2) -> (Batch, Pixels, 2)

            # Campo vectorial 1 (Vertical)
            E1_vec = (self.psi1_base[None, :] * cp.exp(1j * phi1))[:, :, None] * vec_1

            # Campo vectorial 2 (Horizontal)
            E2_vec = (self.psi2_base[None, :] * cp.exp(1j * phi2))[:, :, None] * vec_2

            # 3. Aplicación del Borrador (Si existe)
            if u_eraser is not None:
                # Proyección Geométrica: (E . u) * u
                # Dot product manual sobre el eje de componentes (axis=2)
                # E1 . u
                proj1_mag = E1_vec[:, :, 0] * u_eraser[0] + E1_vec[:, :, 1] * u_eraser[1]
                # E2 . u
                proj2_mag = E2_vec[:, :, 0] * u_eraser[0] + E2_vec[:, :, 1] * u_eraser[1]

                # Reconstruir vectores proyectados
                # Ahora ambos apuntan en la dirección u_eraser
                E1_vec = proj1_mag[:, :, None] * u_eraser[None, None, :]
                E2_vec = proj2_mag[:, :, None] * u_eraser[None, None, :]

            # 4. Superposición y Detección (Resonancia Global)
            # Suma vectorial de campos
            E_total_vec = E1_vec + E2_vec

            # Intensidad = |Ex|^2 + |Ey|^2
            I_batch = cp.abs(E_total_vec[:, :, 0]) ** 2 + cp.abs(E_total_vec[:, :, 1]) ** 2

            I_accum += cp.sum(I_batch, axis=0)

            del E1_vec, E2_vec, E_total_vec, I_batch, phi1, phi2
            if USE_GPU: mempool.free_all_blocks()

            processed += current_batch

        return I_accum / N_SHOTS


# ==========================================
# 5. FUNCIONES AUXILIARES
# ==========================================
def calculate_visibility_robust(I_gpu):
    """Cálculo robusto de visibilidad V = (Imax-Imin)/(Imax+Imin)"""
    if USE_GPU:
        I = cp.asnumpy(I_gpu)
    else:
        I = I_gpu

    if np.max(I) == 0: return 0.0
    I = I / np.max(I)

    # Análisis central
    center_idx = len(I) // 2
    window = len(I) // 10
    I_center = I[center_idx - window: center_idx + window]

    val_max = np.percentile(I_center, 99.9)
    val_min = np.percentile(I_center, 0.1)

    if val_max + val_min == 0: return 0.0
    V = (val_max - val_min) / (val_max + val_min)
    return np.clip(V, 0.0, 1.0)


# ==========================================
# 6. EJECUCIÓN DEL EXPERIMENTO
# ==========================================
if __name__ == "__main__":
    print("-" * 60)
    print(f"INICIANDO SIMULACIÓN DE BORRADOR CUÁNTICO: {SCRIPT_NAME}")
    print(f"Objetivo: Demostrar que la información es ortogonalidad en Lambda.")
    print("-" * 60)

    # 1. Inicializar Motor
    sim = SimEraserEngine(N_PIXELS, SCREEN_WIDTH, SLIT_DIST, SCREEN_DIST, K_WAVE)

    t_start = time.time()

    # ---------------------------------------------------------
    # ESCENARIO A: INFORMACIÓN DE CAMINO (MARCADO)
    # ---------------------------------------------------------
    # Vectores ortogonales |V> y |H>. Sin polarizador.
    print("\n[A] Ejecutando Escenario MARCADO (Vectores Ortogonales)...")
    I_marked_gpu = sim.run_experiment(eraser_angle_deg=None)
    V_marked = calculate_visibility_robust(I_marked_gpu)

    print(f"    -> Visibilidad Observada: {V_marked:.4f}")
    if V_marked < 0.05:
        print("    -> [EXITO] Interferencia destruida por ortogonalidad interna.")
    else:
        print("    -> [FALLO] Interferencia residual inesperada.")

    # ---------------------------------------------------------
    # ESCENARIO B: BORRADO CUÁNTICO (RECUPERACIÓN)
    # ---------------------------------------------------------
    # Proyección a 45 grados. Fuerza a |V> y |H> a proyectarse en |D>.
    # Restauración de indistinguibilidad geométrica.
    print("\n[B] Ejecutando Escenario BORRADO (Proyección a 45 grados)...")
    I_erased_gpu = sim.run_experiment(eraser_angle_deg=45.0)
    V_erased = calculate_visibility_robust(I_erased_gpu)

    print(f"    -> Visibilidad Recuperada: {V_erased:.4f}")
    if V_erased > 0.90:
        print("    -> [EXITO] Interferencia recuperada por proyección geométrica.")
    else:
        print("    -> [FALLO] No se recuperó la interferencia.")

    total_time = time.time() - t_start
    print(f"\n[FIN] Experimento completado en {total_time:.2f} s")

    # ==========================================
    # 7. EXPORTACIÓN DE DATOS
    # ==========================================
    # Transferir a CPU
    if USE_GPU:
        I_marked = cp.asnumpy(I_marked_gpu)
        I_erased = cp.asnumpy(I_erased_gpu)
        x_axis = cp.asnumpy(sim.x_axis) / LAMBDA
    else:
        I_marked = I_marked_gpu
        I_erased = I_erased_gpu
        x_axis = sim.x_axis / LAMBDA

    # Normalizar para comparación justa en gráfica
    I_marked_norm = I_marked / np.max(I_marked)
    I_erased_norm = I_erased / np.max(I_erased)

    df = pd.DataFrame({
        "position_x_lambda": x_axis,
        "intensity_marked": I_marked_norm,
        "intensity_erased": I_erased_norm
    })

    csv_path = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DATA] Datos guardados: {csv_path}")

    # ==========================================
    # 8. GRAFICACIÓN (ONTOLOGÍA VISUAL)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel A: Marcado
    ax1.plot(x_axis, I_marked_norm, 'r-', linewidth=2, label=f'Marcado (Ortogonal) V={V_marked:.2f}')
    ax1.fill_between(x_axis, I_marked_norm, color='red', alpha=0.1)
    ax1.set_title(r"A) Información de Camino Disponible ($|V\rangle \perp |H\rangle$)", fontsize=12)
    ax1.set_ylabel("Intensidad Normalizada", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Texto explicativo en gráfica
    ax1.text(0.02, 0.85, "Suma de Intensidades\nSin términos cruzados", transform=ax1.transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Panel B: Borrado
    ax2.plot(x_axis, I_erased_norm, 'b-', linewidth=1.5, label=f'Borrado (Proyección 45°) V={V_erased:.2f}')
    ax2.fill_between(x_axis, I_erased_norm, color='blue', alpha=0.1)
    ax2.set_title(r"B) Información Borrada (Proyección Geométrica)", fontsize=12)
    ax2.set_xlabel(r"Posición en Pantalla ($x/\lambda$)", fontsize=11)
    ax2.set_ylabel("Intensidad Normalizada", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Texto explicativo
    ax2.text(0.02, 0.85, "Recuperación de Fase\nResonancia Geométrica", transform=ax2.transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.xlim(-150, 150)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[PLOT] Gráfica comparativa guardada en: {plot_path}")
    print("=== PROCESO TERMINADO CON ÉXITO ===")