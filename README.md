# Simulaciones para “Probabilidades y correlaciones cuánticas desde un modelo ondulatorio con espacio interno compacto”  
**Autor:** Miguel J. Saldaña Ramírez  
**Repositorio:** SaldanaRamirez_2025_ProbabilidadesCorrelacionesCuanticas
**Licencia:** MIT

Este repositorio contiene el código fuente y los datos utilizados para generar
las figuras y resultados numéricos del paper:

> Probabilidades y correlaciones cuánticas desde un modelo ondulatorio con espacio interno compacto  
> DOI:

Todas las simulaciones fueron ejecutadas en el entorno `opwaves`.  
Un archivo `environment.yml` está incluido para garantizar reproducibilidad
completa. Para instalaciones simples, se incluye también `requirements.txt`.

---

## 📁 Estructura del repositorio

SaldanaRamirez_2025_ProbabilidadesCorrelacionesCuanticas/
│
├── src/ # Scripts de simulación (Python)
├── data/ # Archivos CSV generados por los scripts
├── requirements.txt # Dependencias básicas (pip)
└── environment.yml # Entorno conda completo (reproducibilidad total)

---

## ▶️ Cómo reproducir las simulaciones

### **1. Crear el entorno Conda (recomendado)**

```bash
conda env create -f environment.yml
conda activate opwaves

pip install -r requirements.txt

---

O instalación mínima con pip

🧪 Scripts incluidos (ordenados según aparición en el paper)
---


Cada script en src/ genera un archivo CSV en la carpeta data/.
Las figuras del paper pueden reconstruirse simplemente ejecutando cada script.

001 — Ley de Malus y Regla de Born emergente  
Script: `001_born_malus.py`  
Datos: `data/001_born_malus_data.csv`  
Fenómeno: Intensidad transmitida vs. ángulo de análisis; validación numérica de la ley de Malus y de la equivalencia con la regla de Born emergente.

002 — Correlaciones CHSH del singlete  
Script: `002_bell_chsh.py`  
Datos: `data/002_chsh_curve_data.csv`  
Fenómeno: Curva de correlación E(θ) del estado tipo singlete, violación de CHSH y verificación de no–señalización en el modelo ondulatorio efectivo.

003 — Rendija doble y decoherencia de fase  
Script: `003_slit.py`  
Datos: `data/003_slit_data.csv`  
Fenómeno: Patrón de interferencia de doble rendija y caída de la visibilidad V(σ) bajo ruido de fase gaussiano, comparada con la predicción teórica \(V(σ)=e^{-σ^2}\).

004 — Borrador cuántico (Quantum Eraser) modelo Classic–Wave  
Script: `004_eraser.py`  
Datos: `data/004_eraser_data.csv`  
Fenómeno: Destrucción y recuperación de interferencias al marcar/borrar información de camino mediante vectores internos ortogonales/proyectados en el espacio Λ.

005 — Scattering (paquete libre + potencial escalón)  
Script: `005_scattering.py`  
Datos: `data/005_free_packet.csv`, `data/005_step_potential.csv`  
Fenómeno: (S1) Dispersión temporal de un paquete gaussiano libre y comparación de σ(t) con la solución analítica; (S3) scattering en un escalón de potencial y cálculo de coeficientes de transmisión/reflexión.

006 — Dinámica efectiva en potencial confinante  
Script: `006_dynamics.py`  
Datos: `data/006_dynamics_stats.csv`  
Fenómeno: Evolución de un paquete desplazado en un oscilador armónico 1D, conservación de norma y energía, y comparación de ⟨x(t)⟩ con la trayectoria clásica.

007 — Diagramas de dispersión efectiva  
Script: `007_dispersion_diagram.py`  
Fenómeno: Cálculo y visualización de las curvas de dispersión ω(k) para la onda base sin masa, el modelo relativista completo y el límite parabólico de Schrödinger.

008 — Espectro emergente de niveles de energía (Apéndice C)  
Script: `008_genesis.py`  
Datos: `data/008_genesis_spectrum.csv`  
Fenómeno: Obtención del espectro de energías mediante la FFT de la autocorrelación temporal de un ensamble de estados en potencial confinante, y comparación de los picos con los niveles teóricos del oscilador armónico.

009 — Correlaciones tripartitas GHZ (validación de Mermin)
Script: `009_multipartita_ghz.py`
Datos: `data/009_datos_correlacion.csv`
Fenómeno: Simulación ondulatoria de correlaciones tripartitas tipo GHZ mediante resonancia global en un espacio interno compacto. Se obtiene el perfil angular
E(0,0,θ) y se maximiza el parámetro de Mermin, verificando la violación del límite clásico M ≤ 2 y alcanzando valores cercanos al límite cuántico M ≈ 4.



---

Descripción de los scripts

..........

001 — Ley de Malus y Regla de Born emergente  
Script: `001_born_malus.py`  
Datos: `data/001_born_malus_data.csv`  
Descripción: Simula un campo interno altamente coherente y calcula, mediante la
integración de amplitudes complejas, la intensidad transmitida por un analizador
de polarización. Verifica numéricamente que la probabilidad de detección
emerge con la ley de Malus \( \cos^2 \), calculando MAE y \(R^2\) frente a la
curva teórica (test directo de la Hipótesis de Born emergente).

.........

002 — Correlaciones CHSH del singlete  
Script: `002_bell_chsh.py`  
Datos: `data/002_chsh_curve_data.csv`  
Descripción: Implementa el singlete isotrópico en el modelo ondulatorio
efectivo. Calcula la correlación \(E(\theta)\) a partir de la resonancia global,
reconstruye la curva \(E(\theta)=-\cos(2\theta)\), evalúa el parámetro CHSH
(obteniendo numéricamente el límite de Tsirelson \(2\sqrt{2}\)) y verifica la
condición de no–señalización en las marginales locales.

.........

003 — Rendija doble y emergencia de la decoherencia  
Script: `003_slit.py`  
Datos: `data/003_slit_data.csv`  
Descripción: Simula una doble rendija escalar con dos caminos ópticos y ruido
de fase gaussiano. Para cada valor de σ calcula la visibilidad de las
interferencias y la compara con la predicción teórica \(V(σ)=e^{-σ^2}\),
mostrando cómo la decoherencia de fase suprime las franjas. Incluye perfiles
coherente/clásico y ajuste cuantitativo (R²) de la curva de decoherencia.

.........

004 — Borrador cuántico (Quantum Eraser) en el modelo Classic–Wave  
Script: `004_eraser.py`  
Datos: `data/004_eraser_data.csv`  
Descripción: Simula un experimento tipo borrador cuántico donde cada rendija
porta un vector interno ortogonal en Λ (información de camino). En el escenario
marcado (sin borrador) la ortogonalidad destruye las interferencias
(Visibilidad ≈ 0). Al introducir un proyector a 45° ambos caminos se proyectan
sobre el mismo estado interno y se recupera la figura de interferencia
(Visibilidad alta). Ilustra que la “información” es ortogonalidad geométrica
en el espacio interno, no transmisión de bits.

.........

005 — Scattering: paquete libre y escalón de potencial  
Script: `005_scattering.py`  
Datos:  
  - `data/005_free_packet.csv`  
  - `data/005_step_potential.csv`  
Descripción: Implementa un motor split–step en 1D para validar la ecuación
efectiva en dos escenarios: (S1) dispersión de un paquete gaussiano libre,
comparando la anchura σ(t) con la solución analítica; (S3) scattering en un
escalón de potencial, calculando coeficientes de transmisión y reflexión y
comparándolos con la fórmula de onda plana. Genera además snapshots de la
densidad y gráficas listas para el paper, verificando unitariedad y consistencia
dinámica del modelo.

.........

006 — Dinámica efectiva en potencial confinante  
Script: `006_dynamics.py`  
Datos: `data/006_dynamics_stats.csv`  
Descripción: Simula la evolución de un paquete gaussiano desplazado en un
oscilador armónico 1D usando un esquema split–step Fourier de alta precisión.
Calcula en el tiempo la norma, la energía total y la posición esperada <x>,
evaluando la deriva numérica (criterio de estabilidad < 1%) y comparando
<x(t)> con la trayectoria clásica x₀ cos(ωt). Proporciona un benchmark directo
de la ecuación efectiva tipo Schrödinger en régimen confinante.

.........

007 — Diagrama de dispersión relativista y límite de Schrödinger  
Script: `007_dispersion_diagram.py`  
Datos: (solo imágenes, no genera CSV)  
Descripción: Calcula y grafica las curvas de dispersión fundamentales del
modelo: (1) onda base sin masa ω = ck, (2) dispersión relativista completa
ω = √(c²k² + ω₀²) con frecuencia de Compton ω₀, y (3) aproximación de
Schrödinger para k → 0. Ilustra cómo emerge el límite parabólico no
relativista y cómo diverge a altas energías. Útil como figura conceptual del
paper (Apéndice A/C).

.........

008 — Espectro emergente (validación de niveles energéticos)  
Script: `008_genesis.py`  
Datos: `data/008_genesis_spectrum.csv`  
Descripción: Genera un ensamble de estados aleatorios ponderados por el 
potencial (“sopa geométrica” de OE) y los evoluciona mediante split–step FFT.
A partir de la autocorrelación temporal obtiene el espectro por FFT, detecta 
los picos y los compara con los niveles teóricos del oscilador armónico 
\(E_n = n + 1/2\). Recupera los primeros 6–7 niveles con error porcentual 
pequeño, validando la capacidad del motor efectivo para reproducir 
cuantización emergente.

009 — Correlaciones tripartitas GHZ y violación de Mermin
Script: `009_multipartita_ghz.py`
Datos: `data/009_datos_correlacion.csv` y `009_reporte_mermin.txt`

Descripción: Implementa la extensión N=3 del mecanismo de resonancia global del
Modelo Ondulatorio Efectivo para generar correlaciones tripartitas análogas al
estado GHZ. El estado interno se modela como una onda estacionaria armónica
Ψ(λ) = cos(3λ), lo que permite interferencia de tercer orden coherente entre
los tres detectores.

1. **Experimento 1 (Perfil angular):**  
   Se fija α = β = 0 y se barre θ ∈ [0, 2π], obteniendo una correlación
   E(0,0,θ) con forma cosenoidal, característica de estados GHZ en mecánica
   cuántica. Los datos crudos se exportan para comparación y graficado.

2. **Experimento 2 (Test de Mermin):**  
   Se optimizan seis ángulos (a₁,a₂,b₁,b₂,c₁,c₂) mediante Nelder–Mead para
   maximizar el parámetro M = |E₁ + E₂ + E₃ − E₄|.  
   El modelo viola el límite clásico M ≤ 2 y alcanza valores cercanos al
   límite cuántico M = 4, reproduciendo la firma GHZ sin introducir
   entrelazamiento explícito en el espacio físico, sino mediante coherencia
   geométrica en Λ.


---
📝 Notas adicionales

Las figuras pueden reconstruirse ejecutando su script correspondiente.

El entorno opwaves del archivo YAML incluye CuPy, NumPy, SciPy y librerías
numéricas avanzadas necesarias para las simulaciones. (CUDA 13)

---
MIT License
---

