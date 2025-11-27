"""
NOMBRE: 006_dynamics.py (Parte 1)
DESCRIPCIÓN: Validación de la Dinámica Efectiva.
             Simulación de alta precisión de la evolución de la envolvente
             ondulatoria en un potencial confinante.
AUTOR: Miguel J. Saldaña - Asistencia (Gemini 3.0 - GTP 5.1)
FECHA: 21 Nov 2025
HARDWARE: GPU Acceleration (CUDA/CuPy)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN DEL ENTORNO
# ==========================================
SCRIPT_NAME = "006_dynamics"
BASE_OUTPUT_DIR = "./data"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, SCRIPT_NAME)

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[SISTEMA] Directorio de resultados: {OUTPUT_DIR}")
except Exception as e:
    print(f"[ERROR] No se pudo crear el directorio: {e}")
    sys.exit(1)

# ==========================================
# 2. GESTIÓN DE HARDWARE (GPU)
# ==========================================
try:
    import cupy as cp

    # Limpieza de memoria agresiva para asegurar estabilidad
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    USE_GPU = True
    print("[HARDWARE] Modo: GPU Acelerada (CuPy) - Alta Precisión")
except ImportError:
    import numpy as cp

    USE_GPU = False
    print("[HARDWARE] Modo: CPU (NumPy) - ADVERTENCIA: Lento para alta resolución.")

# ==========================================
# 3. PARÁMETROS FÍSICOS Y NUMÉRICOS
# ==========================================
# Unidades naturales (hbar = 1, m = 1) típicas en simulaciones fundamentales
HBAR = 1.0
MASS = 1.0

# Parámetros del Potencial (Oscilador Armónico como caso de prueba)
# V(x) = 0.5 * m * omega^2 * x^2
OMEGA = 1.0

# Estado Inicial (Paquete Gaussiano desplazado)
X0 = -2.0  # Posición inicial
P0 = 0.0  # Momento inicial
SIGMA_0 = 1.0  # Anchura inicial

# --- AJUSTE DE PRECISIÓN (Corrección del error del 37%) ---
# El error anterior se debía a un dt muy grande y pocos puntos espaciales.
# Aumentamos la resolución espacial y temporal drásticamente.
N_POINTS = 4096  # Potencia de 2 para FFT rápida
L_BOX = 40.0  # Tamaño de la caja de simulación (-20 a 20)
DT = 0.005  # Paso de tiempo fino (antes era probablemente >0.05)
T_MAX = 2 * np.pi * 2  # Simular 2 periodos completos


# ==========================================
# 4. MOTOR DE DINÁMICA: SPLIT-STEP FOURIER
# ==========================================
class SIMDynamicsEngine:
    """
    Resuelve la Ec.del Paper:
    i hbar d_psi/dt = [-hbar^2/2m nabla^2 + V(x)] psi

    Utiliza el método espectral Split-Step (orden 2) para máxima conservación
    de la unitariedad (norma).
    """

    def __init__(self, n_points, l_box, mass, hbar, dt):
        self.N = n_points
        self.L = l_box
        self.m = mass
        self.hbar = hbar
        self.dt = dt

        # 1. Espacio Real (x)
        self.dx = self.L / self.N
        self.x = cp.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False, dtype=cp.float32)

        # 2. Espacio de Momentos (k)
        # k = 2*pi * freq
        self.k = 2 * cp.pi * cp.fft.fftfreq(self.N, d=self.dx).astype(cp.float32)

        # 3. Operadores de Evolución (Pre-calculados en GPU)
        self._init_operators()

    def _init_operators(self):
        # Operador Cinético (en espacio k): T = p^2 / 2m = (hbar*k)^2 / 2m
        # Evolución: exp(-i T dt / hbar)
        T_op = (self.hbar ** 2 * self.k ** 2) / (2 * self.m)
        self.U_kinetic = cp.exp(-1j * T_op * self.dt / self.hbar).astype(cp.complex64)

        # El Operador Potencial depende de V(x), se define al setear el potencial.
        self.V_x = None
        self.U_potential_half = None

    def set_potential(self, V_array_gpu):
        """Define el potencial externo V(x) y precalcula su operador de evolución."""
        self.V_x = V_array_gpu
        # Evolución medio paso: exp(-i V dt / 2hbar)
        # Usamos medio paso para precisión de 2do orden (Strang Splitting)
        self.U_potential_half = cp.exp(-1j * self.V_x * (self.dt / 2) / self.hbar).astype(cp.complex64)

    def get_initial_state(self, x0, p0, sigma):
        """Genera un paquete gaussiano coherente."""
        # psi ~ exp(i p0 x) * exp(-(x-x0)^2 / 2sigma^2)
        norm = 1.0 / (np.pi ** (0.25) * cp.sqrt(sigma))
        psi = norm * cp.exp(1j * p0 * self.x / self.hbar) * \
              cp.exp(-(self.x - x0) ** 2 / (2 * sigma ** 2))
        return psi.astype(cp.complex64)

    def step(self, psi):
        """
        Avanza el estado un paso temporal dt usando Split-Step:
        psi(t+dt) = U_V(dt/2) * IFFT * U_T(dt) * FFT * U_V(dt/2) * psi(t)
        """
        # 1. Medio paso de Potencial
        psi = self.U_potential_half * psi

        # 2. Paso completo Cinético (en espacio de frecuencias)
        psi_k = cp.fft.fft(psi)
        psi_k = self.U_kinetic * psi_k
        psi = cp.fft.ifft(psi_k)

        # 3. Medio paso de Potencial
        psi = self.U_potential_half * psi

        return psi


# ==========================================
# 5. FUNCIONES AUXILIARES (OBSERVABLES)
# ==========================================
def calculate_observables(psi, x_gpu, V_gpu, engine):
    """
    Calcula Norma, Posición Esperada y Energía Total.
    Validación crítica de la estabilidad numérica de la propuesta.
    """
    dx = engine.dx

    # 1. Densidad y Norma
    density = cp.abs(psi) ** 2
    norm = cp.sum(density) * dx

    # 2. Posición Esperada <x>
    # Normalizamos por si acaso hay una deriva infinitesimal en la norma
    x_mean = cp.sum(x_gpu * density) * dx / norm

    # 3. Energía Total <H> = <T> + <V>
    # <V> = Integral( psi* V psi )
    E_pot = cp.sum(density * V_gpu) * dx

    # <T> = Integral( psi* (-h^2/2m d^2/dx^2) psi )
    # Lo calculamos espectralmente: <T> = Sum( |psi_k|^2 * (hk)^2/2m )
    psi_k = cp.fft.fft(psi)
    # Ajuste de normalización de Parseval para FFT discreta en CuPy
    # Integral |psi|^2 dx = 1  <--> Sum |psi_k|^2 dk = 2pi
    # Forma más segura: T_op aplicado a psi
    T_op_k = (engine.hbar ** 2 * engine.k ** 2) / (2 * engine.m)
    T_psi_k = T_op_k * psi_k
    T_psi = cp.fft.ifft(T_psi_k)

    # <T> = Integral( psi* . T_psi )
    E_kin = cp.sum(cp.conj(psi) * T_psi).real * dx

    total_energy = (E_pot + E_kin) / norm

    return float(norm), float(x_mean), float(total_energy)


# ==========================================
# 6. BLOQUE PRINCIPAL DE EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    print("-" * 60)
    print(f"INICIANDO VALIDACIÓN DINÁMICA: {SCRIPT_NAME}")
    print(f"Objetivo: Estabilidad de Ec. Efectiva")
    print("-" * 60)

    # 1. Inicializar Motor
    sim = SIMDynamicsEngine(N_POINTS, L_BOX, MASS, HBAR, DT)

    # 2. Definir Potencial (Oscilador Armónico)
    # V(x) = 0.5 * m * w^2 * x^2
    V_gpu = 0.5 * MASS * (OMEGA ** 2) * (sim.x ** 2)
    sim.set_potential(V_gpu)

    # 3. Estado Inicial (Coherente desplazado)
    psi = sim.get_initial_state(X0, P0, SIGMA_0)

    # Listas para historial (CPU)
    times = []
    norms = []
    positions = []
    energies = []

    # Guardar snapshots para animación/gráfica (Inicial, 1/4, 1/2, Final)
    snapshots_psi = []
    snapshots_t = []

    # Configuración de pasos
    n_steps = int(T_MAX / DT)
    save_interval = n_steps // 50  # Guardar 50 puntos de datos para CSV
    snapshot_indices = [0, n_steps // 4, n_steps // 2, n_steps - 1]

    print(f"[PROGRESO] Simulando {n_steps} pasos temporales (dt={DT})...")
    t_start = time.time()

    # --- BUCLE TEMPORAL ---
    for step in range(n_steps):
        # A. Guardar Datos
        if step % save_interval == 0 or step in snapshot_indices:
            n_val, x_val, e_val = calculate_observables(psi, sim.x, V_gpu, sim)
            times.append(step * DT)
            norms.append(n_val)
            positions.append(x_val)
            energies.append(e_val)

            # Guardar perfil de onda para gráfica
            if step in snapshot_indices:
                if USE_GPU:
                    snapshots_psi.append(cp.asnumpy(cp.abs(psi) ** 2))
                else:
                    snapshots_psi.append(np.abs(psi) ** 2)
                snapshots_t.append(step * DT)

        # B. Evolución Unitaria
        psi = sim.step(psi)

    total_time = time.time() - t_start
    print(f"[FIN] Simulación completada en {total_time:.2f} s")

    # ==========================================
    # 7. ANÁLISIS DE ESTABILIDAD
    # ==========================================
    # Cálculo de derivas (errores numéricos)
    delta_norm = abs(norms[-1] - norms[0])
    delta_energy = abs(energies[-1] - energies[0]) / energies[0]

    print("-" * 40)
    print("REPORTE DE ESTABILIDAD")
    print("-" * 40)
    print(f"Norma Inicial: {norms[0]:.6f} | Final: {norms[-1]:.6f}")
    print(f"Deriva de Norma: {delta_norm:.2e} (Debe ser < 1e-4)")
    print(f"Energía Inicial: {energies[0]:.6f} | Final: {energies[-1]:.6f}")
    print(f"Error Relativo Energía: {delta_energy * 100:.4f}%")

    valid_flag = "EXITOSO" if delta_energy < 0.01 else "FALLIDO"  # Criterio 1%
    print(f"VEREDICTO TÉCNICO: {valid_flag}")
    print("-" * 40)

    # ==========================================
    # 8. EXPORTACIÓN DE DATOS
    # ==========================================
    import pandas as pd

    df = pd.DataFrame({
        "time": times,
        "norm": norms,
        "position_x": positions,
        "total_energy": energies
    })

    csv_path = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DATA] Estadísticas guardadas: {csv_path}")

    # ==========================================
    # 9. GRAFICACIÓN
    # ==========================================
    # Recuperar eje x
    if USE_GPU:
        x_cpu = cp.asnumpy(sim.x)
        V_cpu = cp.asnumpy(V_gpu)
    else:
        x_cpu = sim.x
        V_cpu = V_gpu

    # --- Gráfica A: Evolución del Paquete ---
    plt.figure(figsize=(10, 6))

    # Graficar Potencial (Escalado para visibilidad)
    scale_factor = np.max(snapshots_psi[0]) / np.max(V_cpu) * 0.5
    plt.plot(x_cpu, V_cpu * scale_factor, 'k--', alpha=0.5, label='Potencial V(x)')

    # Graficar Snapshots
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['t=0 (Inicio)', 't=T/4', 't=T/2', 't=Final']

    for i, dens in enumerate(snapshots_psi):
        plt.plot(x_cpu, dens, color=colors[i], linewidth=1.5, alpha=0.8, label=labels[i])
        # Llenar área para efecto visual
        plt.fill_between(x_cpu, dens, alpha=0.1, color=colors[i])

    plt.title(f"Dinámica Efectiva Sim: Oscilador Armónico\nValidación Ec. Schrödinger Emergente", fontsize=12)
    plt.xlabel("Posición x", fontsize=11)
    plt.ylabel("Densidad de Probabilidad $|\psi|^2$", fontsize=11)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(-8, 8)

    plot_path_A = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_evolution.png")
    plt.savefig(plot_path_A, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Gráfica B: Conservación (Prueba de Fuego) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Energía
    ax1.plot(times, energies, 'r-', linewidth=1.5)
    ax1.set_ylabel("Energía Total <H>")
    ax1.set_title(f"Conservación de Energía (Error: {delta_energy * 100:.3f}%)")
    ax1.grid(True, alpha=0.3)

    # Trayectoria (Teoría Clásica vs Simulación)
    # x(t) teórico = x0 cos(wt)
    t_arr = np.array(times)
    x_teo = X0 * np.cos(OMEGA * t_arr)

    ax2.plot(t_arr, positions, 'b-', label='Simulación (<x>)')
    ax2.plot(t_arr, x_teo, 'k:', label='Teoría Clásica')
    ax2.set_ylabel("Posición Esperada <x>")
    ax2.set_xlabel("Tiempo")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plot_path_B = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_conservation.png")
    plt.savefig(plot_path_B, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Gráficas generadas en: {OUTPUT_DIR}")
    print("=== PROCESO TERMINADO CON ÉXITO ===")