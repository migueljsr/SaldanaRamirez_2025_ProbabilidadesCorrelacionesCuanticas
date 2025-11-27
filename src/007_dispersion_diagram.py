"""
NOMBRE: 007_dispersion_diagram.py
AUTOR: Miguel J. Saldaña - Asistencia (Gemini 3.0 - GTP 5.1)
HARDWARE: GPU Acceleration (CUDA/CuPy)
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def generar_diagrama_dispersion():
    # -------------------------------------------------------------------------
    # 1. Configuración de Directorios y Archivos
    # -------------------------------------------------------------------------
    # Definimos la ruta base solicitada
    base_dir = "./data"
    script_name = "007_dispersion_diagram"

    # Creamos la carpeta específica para este resultado
    output_folder = os.path.join(base_dir, script_name)

    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Directorio creado: {output_folder}")
        else:
            print(f"El directorio ya existe: {output_folder}")
    except OSError as e:
        print(f"Error al crear directorios. Verifica permisos o ruta. {e}")
        return

    # -------------------------------------------------------------------------
    # 2. Definición de Parámetros Físicos (Unidades Arbitrarias para Visualización)
    # -------------------------------------------------------------------------
    c = 1.0  # Velocidad de la luz (referencia pendiente = 1)
    w0 = 2.0  # Frecuencia de Compton (Masa en reposo / hbar)

    # Rango de número de onda k (momento)
    k = np.linspace(0, 5, 500)

    # -------------------------------------------------------------------------
    # 3. Cálculo de las Curvas de Dispersión
    # -------------------------------------------------------------------------

    # A. Onda Primordial / Luz (Referencia asintótica)
    # omega = c * k
    w_light = c * k

    # B. Dispersión Relativista Completa (E^2 = p^2c^2 + m^2c^4)
    # omega = sqrt(c^2 k^2 + w0^2)
    # Esta es la curva real que obedece una onda confinada o con masa efectiva.
    w_rel = np.sqrt(c ** 2 * k ** 2 + w0 ** 2)

    # C. Aproximación de Schrödinger (Régimen No Relativista)
    # Expansión de Taylor de w_rel alrededor de k=0:
    # omega approx w0 + (c^2 k^2) / (2 w0)
    # Esta es la parábola que derivamos en el Apéndice A.
    w_sch = w0 + (c ** 2 * k ** 2) / (2 * w0)

    # -------------------------------------------------------------------------
    # 4. Generación del Gráfico
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 7), dpi=300)

    # Estilo general
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

    # Plotear curvas
    # 1. Línea de luz (cono de luz)
    plt.plot(k, w_light, 'k--', linewidth=1.5, alpha=0.5, label=r'Onda Base (c)')

    # 2. Relativista (Hiperbola)
    plt.plot(k, w_rel, 'b-', linewidth=2.5, label=r'Dispersión Efectiva (Relativista)')

    # 3. Schrödinger (Parábola)
    plt.plot(k, w_sch, 'r-.', linewidth=2.0, label=r'Aproximación Schrödinger (No Rel.)')

    # -------------------------------------------------------------------------
    # 5. Anotaciones y Detalles Conceptuales
    # -------------------------------------------------------------------------

    # Marca de la Frecuencia de Reposo (Masa)
    plt.scatter([0], [w0], color='black', s=50, zorder=5)
    plt.text(0.1, w0, r'$\omega_0 = mc^2/\hbar$', fontsize=12, verticalalignment='bottom')

    # Región de validez de Schrödinger (Zoom k -> 0)
    plt.fill_between(k[:150], w_rel[:150], w_sch[:150], color='gray', alpha=0.1)
    plt.annotate('Régimen No Relativista\n(Coincidencia Parabólica)',
                 xy=(1.0, w_rel[100]), xytext=(2.0, 1.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10)

    # Divergencia a altas energías
    plt.annotate('Desviación a alta energía',
                 xy=(4.0, w_sch[400]), xytext=(3.0, 6.0),
                 fontsize=10)

    # Etiquetas y Título
    plt.xlabel(r'Número de onda $k$ (Momento)', fontsize=12)
    plt.ylabel(r'Frecuencia $\omega$ (Energía)', fontsize=12)
    plt.title(r'Emergencia Dinámica: De la Portadora a Schrödinger', fontsize=14, pad=15)

    plt.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Límites para mantener la estética
    plt.xlim(0, 5)
    plt.ylim(0, 8)

    # -------------------------------------------------------------------------
    # 6. Guardado
    # -------------------------------------------------------------------------
    image_name = f"{script_name}.png"
    save_path = os.path.join(output_folder, image_name)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Gráfico generado exitosamente en: {save_path}")

    # Guardar una copia de este mismo script en la carpeta de destino (autoreplicación solicitada)
    # Leemos el contenido del archivo actual
    try:
        current_script_path = os.path.abspath(__file__)  # Funciona si se ejecuta desde archivo
        with open(current_script_path, 'r', encoding='utf-8') as src:
            content = src.read()

        target_script_path = os.path.join(output_folder, f"{script_name}.py")
        with open(target_script_path, 'w', encoding='utf-8') as dst:
            dst.write(content)
        print(f"Script guardado en: {target_script_path}")
    except NameError:
        # En caso de ejecutarse en entornos interactivos (notebooks) donde __file__ no existe
        print("Nota: Si estás ejecutando esto en un notebook, guarda este código manualmente en la ruta indicada.")


if __name__ == "__main__":
    generar_diagrama_dispersion()