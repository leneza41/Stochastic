# Manual de Ejecución y Prerrequisitos en Windows 11

**Proyecto:** Evaluación del Rendimiento y Convergencia de Algoritmos de Búsqueda Incremental (**D\* Lite**, **LPA\***, **Incremental A\***) ante Fallos Estocásticos de Aristas en Tiempo Real  
**Autores:** Leonardo Núñez Zavala, Alan Yael Fonseca Ruiz, Roberto Gordillo Herrera

---

## 1. Requisitos del Sistema (Priorizado para Windows 11)

| Componente | Requisito Mínimo | Configuración Recomendada (Windows 11) |
| :--- | :--- | :--- |
| **Sistema Operativo** | Windows 10 (64-bit) | **Windows 11 (64-bit, Build 22621 / 26100 o superior)** |
| **Terminal / Shell** | PowerShell 5.1 o CMD | **Windows Terminal con PowerShell 7+ o Windows PowerShell** |
| **Intérprete Python** | Python 3.10+ (64-bit) | **Python 3.11 – 3.15 (64-bit)** agregado al `PATH` |
| **Compilador C++ (Opcional)** | C++17 compatible | **MSVC (Visual Studio 2022 / 2026 Build Tools)** + **CMake 3.16+** |
| **Memoria RAM** | 4 GB | **8 GB o más** |

> **Nota clave sobre dependencias en Windows 11:**  
> El pipeline principal (`python/run_research_pipeline.py` y `run_experiments.ps1`) está diseñado para funcionar **de inmediato en Windows 11 incluso sin paquetes externos instalados** (*zero-dependency mode*), ya que invoca directamente las APIs nativas Win32 (`K32GetProcessMemoryInfo` y `GlobalMemoryStatusEx` mediante `ctypes.windll.kernel32`) para medir la memoria RAM residente del proceso (RSS) y genera gráficos vectoriales `.svg` y tableros `.html`/`.md` de calidad de publicación. Opcionalmente pueden instalarse `numpy`, `pandas`, `scipy`, `osmnx` y `psutil` desde `requirements.txt`.

---

## 2. Preparación del Entorno en Windows 11

### 2.1. Abrir PowerShell en la Carpeta del Proyecto
1. Abre **Windows Terminal** o **PowerShell** en Windows 11.
2. Navega al directorio raíz del repositorio:
   ```powershell
   cd C:\Documents\ITSUR\TI\Stochastic
   ```
3. Verifica que Python esté disponible:
   ```powershell
   python --version
   ```

### 2.2. (Opcional) Crear un Entorno Virtual e Instalar Paquetes Complementarios
Si deseas instalar las bibliotecas opcionales listadas en `requirements.txt`:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
*(Si PowerShell bloquea la ejecución de scripts en tu cuenta de usuario, habilita temporalmente la política para el proceso actual ejecutando: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`).*

---

## 3. Cómo Ejecutar el Pipeline de Resultados Especificando la Carpeta Destino (Pasos 3 y 4)

Puedes ejecutar todo el experimento (preparación de grafos de **OpenStreetMap**, **Moving AI** y **sintéticos**, corridas de calentamiento descartadas, medición de tiempo en ms, nodos recalculados y memoria en MB, pruebas estadísticas de **Friedman** y **Wilcoxon con ajuste de Holm**, gráficas vectoriales y reporte académico) pasando la **ruta de la carpeta destino como argumento**.

### Opción A: Usando el Script de Python Directamente (Recomendado)

```powershell
# Sintaxis pasando la ruta destino como primer argumento posicional:
python python\run_research_pipeline.py "C:\Ruta\Destino\Resultados_Investigacion"
```

También puedes personalizar el número de corridas por mapa (`--runs-per-map`, p. ej. `30` para la muestra piloto completa establecida en *Muestra de la Investigación*) y la cantidad de fallos estocásticos secuenciales por corrida (`--failures-per-run`):

```powershell
# Ejecución con muestra piloto completa de 30 corridas por mapa y 10 fallos estocásticos por corrida:
python python\run_research_pipeline.py "C:\Ruta\Destino\Resultados_Piloto30" --runs-per-map 30 --failures-per-run 10
```

### Opción B: Usando el Script Automatizado de PowerShell para Windows 11 (`run_experiments.ps1`)

```powershell
# Permitir ejecución en la sesión actual de PowerShell (si es necesario):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Ejecutar indicando la carpeta destino como argumento:
.\run_experiments.ps1 "C:\Ruta\Destino\Resultados_Investigacion"

# O especificando parámetros adicionales:
.\run_experiments.ps1 -OutputDir "C:\Ruta\Destino\Resultados_Piloto30" -RunsPerMap 30 -FailuresPerRun 10
```

---

## 4. Estructura de la Carpeta Destino Generada

Al finalizar la ejecución, la carpeta destino que especificaste contendrá automáticamente la siguiente estructura organizada en estilo de investigación científica:

```text
<Carpeta_Destino>/
├── 01_raw_data/
│   ├── replanning_steps_raw.csv                       # Mediciones crudas paso a paso tras cada fallo estocástico
│   ├── experimental_runs_raw.csv                      # Unidades muestrales (promedios por corrida completa)
│   └── system_environment.json                        # Registro de hardware, CPU, RAM y versión de Windows 11
├── 02_statistical_tables/
│   ├── table1_descriptive_by_algorithm_and_source.csv # Media, DE, Mediana, RIC, P95 y Outliers por fuente
│   ├── table2_performance_by_failure_mode.csv         # Desempeño por modo de fallo estocástico
│   ├── table3_scale_sensitivity_analysis.csv          # Contraste de escala (Pequeña vs. Mediana vs. Grande)
│   ├── table4_sample_size_estimation.csv              # Cálculo muestral Z=1.96 (Población infinita y finita)
│   ├── table5_friedman_wilcoxon_tests.csv             # Prueba de Friedman, W de Kendall y Wilcoxon-Holm
│   └── table6_efficiency_resource_tradeoff.csv        # Índice compuesto de equilibrio eficiencia-recursos
├── 03_figures/
│   ├── fig1_replan_time_comparison.svg                # Gráfica 1: Tiempo de cómputo (ms) por fuente de grafo
│   ├── fig2_nodes_recalculated_comparison.svg         # Gráfica 2: Nodos recalculados por fuente de grafo
│   ├── fig3_memory_consumption_comparison.svg         # Gráfica 3: Memoria estructural (KB) por algoritmo
│   ├── fig4_convergence_over_failure_sequence.svg     # Gráfica 4: Convergencia a lo largo de los fallos 1..K
│   ├── fig5_scale_effect_small_to_large.svg           # Gráfica 5: Sensibilidad al escalamiento del grafo
│   └── fig6_efficiency_memory_tradeoff_scatter.svg    # Gráfica 6: Plano de equilibrio Tiempo vs. Memoria y Nodos
└── 04_research_report/
    ├── RESEARCH_REPORT.md                             # Informe científico completo en formato Markdown
    └── RESEARCH_DASHBOARD.html                        # Tablero interactivo autocontenido (abrir en navegador)
```

### Cómo Visualizar el Reporte Interactivo en Windows 11
Desde PowerShell, ejecuta:
```powershell
Start-Process "C:\Ruta\Destino\Resultados_Investigacion\04_research_report\RESEARCH_DASHBOARD.html"
```
Esto abrirá el informe académico completo con todas las tablas estadísticas, pruebas de hipótesis y figuras vectoriales incrustadas directamente en Microsoft Edge / Google Chrome.

---

## 5. Compilación y Ejecución del Motor en C++17 en Windows 11 (MSVC / CMake)

Si deseas compilar y ejecutar también el motor nativo en **C++17** bajo Windows 11:

1. Abre **Developer PowerShell for VS 2022 / 2026** (o una terminal con `cmake` y `cl.exe` en el `PATH`).
2. Ejecuta:
   ```powershell
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```
3. Ejecuta las pruebas de equivalencia y simulación en C++:
   ```powershell
   ctest --output-on-failure -C Release
   ```
4. Ejecuta el generador de resultados en C++ indicando la carpeta destino como argumento:
   ```powershell
   .\Release\run_benchmark_results.exe "C:\Ruta\Destino\Resultados_CPP" 10
   ```
