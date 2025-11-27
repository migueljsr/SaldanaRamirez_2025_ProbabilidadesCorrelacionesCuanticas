"""
SCRIPT: 009_multipartita_ghz.py
DESCRIPCIÓN: Simulación numérica de correlaciones tripartitas (GHZ) y verificación
de la desigualdad de Mermin utilizando el Modelo Ondulatorio Efectivo.
AUTOR: Miguel Junior Saldaña Ramírez - Asistencia (Gemini 3.0 - GTP 5.1)
FECHA: Noviembre 2025
"""

import os
import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# =============================================================================
# 1. CONFIGURACIÓN DEL ENTORNO
# =============================================================================

# Ruta base del proyecto
BASE_PATH = "./data"
PROJECT_NAME = "009_multipartita"
OUTPUT_DIR = os.path.join(BASE_PATH, PROJECT_NAME)

# Verificación de directorio
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Parámetros de simulación
N_SAMPLES = 10 ** 7  # Resolución de integración en el espacio interno
BATCH_SIZE = 10 ** 6  # Tamaño de lote para gestión de memoria VRAM

print(f"[SISTEMA] Inicializando entorno de simulación en GPU...")
print(f"[SISTEMA] Dispositivo: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')}")
print(f"[SISTEMA] Resolución de espacio interno: {N_SAMPLES:.1e} puntos")


# =============================================================================
# 2. DEFINICIÓN DEL MODELO ONDULATORIO (CLASES)
# =============================================================================

class InternalSpace:
    """
    Representación discretizada del espacio interno compacto Lambda = [0, 2pi).
    """

    def __init__(self, n_samples):
        self.n = n_samples
        # Generación del dominio angular en GPU
        self.lambda_space = cp.linspace(0, 2 * cp.pi, n_samples, dtype=cp.float32)
        # Medida invariante normalizada d_mu
        self.d_mu = 1.0 / (2 * cp.pi)


class WaveFunction:
    """
    Estado físico del sistema definido sobre el espacio interno.
    Para la configuración GHZ, se utiliza un estado estacionario armónico
    que permite interferencia constructiva en la intensidad de tercer orden.
    """

    def __init__(self, internal_space):
        self.space = internal_space
        # Estado estacionario: Psi(lambda) = cos(3*lambda)
        # La frecuencia armónica 3 permite resonancia tripartita simétrica.
        self.psi = cp.cos(3 * self.space.lambda_space)

    # =============================================================================


# 3. MOTOR DE CÁLCULO (RESONANCIA GLOBAL)
# =============================================================================

def get_amplitude_batch(psi, space, angle_A, angle_B, angle_C, signs):
    """
    Calcula la amplitud de resonancia global A_abc para una configuración dada.

    Args:
        psi: Objeto WaveFunction.
        space: Objeto InternalSpace.
        angle_A, angle_B, angle_C: Ajustes macroscópicos de los detectores.
        signs: Tupla (s_a, s_b, s_c) donde +1 indica cos(L-a) y -1 indica sin(L-a).

    Returns:
        Amplitud escalar compleja integrada sobre Lambda.
    """
    lam = space.lambda_space

    # Selección de la base de proyección geométrica según el canal (+/-)
    P_a = cp.cos(lam - angle_A) if signs[0] == 1 else cp.sin(lam - angle_A)
    P_b = cp.cos(lam - angle_B) if signs[1] == 1 else cp.sin(lam - angle_B)
    P_c = cp.cos(lam - angle_C) if signs[2] == 1 else cp.sin(lam - angle_C)

    # Integrando del funcional de resonancia
    integrand = psi.psi * P_a * P_b * P_c

    # Factor de escala geométrico para la dimensionalidad N=3
    scale_factor = 4.0

    return cp.sum(integrand) * space.d_mu * scale_factor


def calculate_E_value(psi, space, alpha, beta, gamma):
    """
    Calcula el valor esperado de correlación E(a,b,c) para ajustes fijos.
    Implementa la Regla de Born efectiva mediante normalización de intensidades.
    """
    # Combinaciones de resultados posibles (a, b, c) en {+1, -1}^3
    outcomes = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
    ]

    intensities = []
    product_values = []

    # Cálculo de intensidad para cada canal de salida
    for (sa, sb, sc) in outcomes:
        # 1. Amplitud Global
        Amp = get_amplitude_batch(psi, space, alpha, beta, gamma, (sa, sb, sc))
        # 2. Intensidad (Regla de Born pre-normalizada)
        I = cp.abs(Amp) ** 2

        intensities.append(I)
        product_values.append(sa * sb * sc)  # Producto de resultados (+1 o -1)

    intensities = cp.array(intensities)
    total_intensity = cp.sum(intensities)

    # Evitar división por cero numérico
    if total_intensity < 1e-9:
        return 0.0

    # 3. Normalización (Probabilidades observables)
    probabilities = intensities / total_intensity

    # 4. Valor Esperado E = Sum( a*b*c * P(a,b,c) )
    E = cp.sum(cp.array(product_values) * probabilities)

    return float(E)

# =============================================================================
# 4. BLOQUE PRINCIPAL DE EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print(" SIMULACIÓN DE CORRELACIONES MULTIPARTITAS (ESTADO GHZ)")
        print(" MODELO ONDULATORIO EFECTIVO")
        print("=" * 60)

        # -------------------------------------------------------------------------
        # A. Inicialización del Sistema
        # -------------------------------------------------------------------------
        t_init = time.time()

        # 1. Construcción del Espacio Interno
        space = InternalSpace(N_SAMPLES)

        # 2. Preparación del Estado Físico
        # Se utiliza el modo armónico cos(3*lambda) para generar interferencia de intensidad
        # en la integral triple, análogo a la coherencia del estado GHZ.
        psi = WaveFunction(space)

        print(f"[SETUP] Estado cargado en VRAM. Configuración: GHZ (Armónico 3)")

        # -------------------------------------------------------------------------
        # B. Experimento 1: Perfil de Correlación Angular
        # -------------------------------------------------------------------------
        print("\n[EXP 1] Obteniendo perfil de correlación E(0, 0, theta)...")

        n_steps = 60
        theta_range = np.linspace(0, 2 * np.pi, n_steps)
        correlations = []

        # Barrido: A=0, B=0, C rota de 0 a 2pi
        for th in theta_range:
            E_val = calculate_E_value(psi, space, 0.0, 0.0, th)
            correlations.append(E_val)

        # Exportar datos crudos
        df_results = pd.DataFrame({
            "Theta_C_rad": theta_range,
            "Correlation_E": correlations
        })
        csv_path = os.path.join(OUTPUT_DIR, "009_datos_correlacion.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"[DATOS] Perfil angular guardado en: {csv_path}")

        # -------------------------------------------------------------------------
        # C. Experimento 2: Test de Desigualdad de Mermin
        # -------------------------------------------------------------------------
        print("\n[EXP 2] Verificando violación de Desigualdad de Mermin...")
        print("        Criterio: M_clásico <= 2.0  |  M_cuántico <= 4.0")


        # Función objetivo para maximizar M = |E1 + E2 + E3 - E4|
        # Vector de parámetros x = [a1, a2, b1, b2, c1, c2]
        def mermin_objective(x):
            a1, a2, b1, b2, c1, c2 = x
            # Términos de la desigualdad
            E1 = calculate_E_value(psi, space, a1, b1, c2)
            E2 = calculate_E_value(psi, space, a1, b2, c1)
            E3 = calculate_E_value(psi, space, a2, b1, c1)
            E4 = calculate_E_value(psi, space, a2, b2, c2)

            # Queremos maximizar la magnitud de la combinación lineal
            return -abs(E1 + E2 + E3 - E4)


        # Semilla de ángulos (configuración estándar GHZ)
        x_initial = [0, np.pi / 2, 0, np.pi / 2, 0, np.pi / 2]

        # Optimización numérica
        result = minimize(mermin_objective, x_initial, method='Nelder-Mead', tol=1e-3)
        M_max = -result.fun

        # Reporte de resultados en consola
        print("-" * 45)
        print(f" RESULTADO FINAL (Parámetro M): {M_max:.5f}")
        print("-" * 45)

        if M_max > 2.0:
            print(f">>> VIOLACIÓN CONFIRMADA: El modelo supera el límite local.")
            print(f">>> Exceso sobre límite clásico: +{(M_max - 2.0):.4f}")
        else:
            print(">>> RESULTADO CLÁSICO: No se observa violación.")

        # Guardar reporte detallado
        txt_path = os.path.join(OUTPUT_DIR, "009_reporte_mermin.txt")
        with open(txt_path, "w") as f:
            f.write("REPORTE DE SIMULACION: TEST DE MERMIN (GHZ)\n")
            f.write("===========================================\n")
            f.write(f"Estado Interno: Onda Estacionaria cos(3*lambda)\n")
            f.write(f"Resolución Espacial: {N_SAMPLES}\n")
            f.write(f"Valor M Obtenido: {M_max:.5f}\n")
            f.write(f"Límite Local Realista: 2.0\n")
            f.write(f"Conclusión: {'Modelo No-Local' if M_max > 2 else 'Modelo Local'}\n\n")
            f.write(f"Ángulos Óptimos (rad):\n{result.x}\n")

        # -------------------------------------------------------------------------
        # D. Visualización de Resultados
        # -------------------------------------------------------------------------
        plt.figure(figsize=(10, 6))

        # Estilo académico sobrio
        plt.plot(theta_range, correlations, 'o-', color='#4B0082', linewidth=1.5, markersize=4,
                 label='Modelo Ondulatorio (Simulación)')

        # Líneas de referencia
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.axhline(y=-1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        plt.title(
            f"Correlación Tripartita GHZ: $E(\\alpha, \\beta, \\theta)$ [$\\alpha=\\beta=0$]\nParámetro Mermin $M \\approx {M_max:.3f}$",
            fontsize=11)
        plt.xlabel(r"Ángulo Detector C ($\theta$) [rad]", fontsize=10)
        plt.ylabel(r"Correlación $E$", fontsize=10)
        plt.legend(frameon=True, fancybox=False, edgecolor='black')
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.tight_layout()

        img_path = os.path.join(OUTPUT_DIR, "009_grafico_ghz.png")
        plt.savefig(img_path, dpi=300)
        print(f"[GRAFICO] Imagen guardada en: {img_path}")

        print(f"\n[SISTEMA] Proceso finalizado exitosamente en {time.time() - t_init:.2f}s.")

    except Exception as e:
        print(f"\n[ERROR] Se produjo una excepción durante la ejecución:\n{e}")