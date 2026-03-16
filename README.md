# Predicción del Rendimiento Académico en las Pruebas Saber Pro

> **Competencia Kaggle:** [UDEA AI 4 Eng 20252 — Pruebas Saber Pro Colombia](https://www.kaggle.com/competitions/udea-ai-4-eng-20252-pruebas-saber-pro-colombia)

---

## Descripción del Proyecto

Las **Pruebas Saber Pro** (ECAES) son el examen de Estado que aplica el ICFES a los estudiantes próximos a graduarse de programas de educación superior en Colombia. Evalúan competencias genéricas (lectura crítica, razonamiento cuantitativo, competencias ciudadanas, comunicación escrita e inglés) y específicas por área de conocimiento.

Este proyecto construye un modelo de clasificación para **predecir el nivel de rendimiento global** de un estudiante a partir de sus características socioeconómicas, académicas y demográficas, sin usar los puntajes directos del examen.

---

## El Problema

**Tarea:** Clasificación multiclase con 4 categorías ordenadas.

| Clase | Descripción |
|-------|-------------|
| `bajo` | Rendimiento bajo |
| `medio-bajo` | Rendimiento medio-bajo |
| `medio-alto` | Rendimiento medio-alto |
| `alto` | Rendimiento alto |

El dataset está **balanceado** (~173,000 estudiantes por clase), lo que hace que el baseline aleatorio sea del 25%.

---

## Dataset

**Tamaño:** 692,500 registros × 21 variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `PERIODO_ACADEMICO` | Temporal | Período de presentación del examen |
| `E_PRGM_ACADEMICO` | Categórica | Programa académico cursado |
| `E_PRGM_DEPARTAMENTO` | Categórica | Departamento de la institución |
| `E_VALORMATRICULAUNIVERSIDAD` | Ordinal | Valor de la matrícula (8 rangos) |
| `E_HORASSEMANATRABAJA` | Ordinal | Horas semanales de trabajo (5 niveles) |
| `F_ESTRATOVIVIENDA` | Ordinal | Estrato socioeconómico (1–6) |
| `F_TIENEINTERNET` | Binaria | Acceso a internet en casa |
| `F_TIENECOMPUTADOR` | Binaria | Tiene computador en casa |
| `F_TIENELAVADORA` | Binaria | Tiene lavadora |
| `F_TIENEAUTOMOVIL` | Binaria | Tiene automóvil |
| `F_EDUCACIONPADRE` | Ordinal | Nivel educativo del padre (0–9) |
| `F_EDUCACIONMADRE` | Ordinal | Nivel educativo de la madre (0–9) |
| `E_PAGOMATRICULAPROPIO` | Binaria | Paga matrícula con recursos propios |
| `INDICADOR_1` | Continua | Competencia genérica 1 |
| `INDICADOR_2` | Continua | Competencia genérica 2 |
| `INDICADOR_3` | Continua | Competencia genérica 3 |
| `INDICADOR_4` | Continua | Competencia genérica 4 |
| `RENDIMIENTO_GLOBAL` | **Target** | Rendimiento global (4 clases) |

> Los indicadores se excluyen del entrenamiento para evitar fuga de datos (*data leakage*), ya que son componentes directos del rendimiento predicho.

---

## Metodología

```
Datos Kaggle
    │
    ▼
01 - EDA ──────────────── Exploración, calidad, patrones, visualizaciones
    │
    ▼
02 - Preprocesado ──────── Codificación ordinal, binaria y de target
    │
    ▼
Feature Engineering ────── 45+ nuevas variables (agregados, ratios, interacciones)
    │
    ├──▶ 03 - Random Forest
    ├──▶ 04 - XGBoost
    └──▶ 99 - CatBoost (solución final)
```

### Ingeniería de Características

Se construyeron **45+ features** adicionales organizados en 7 grupos:

| Grupo | Ejemplos | Justificación |
|-------|----------|---------------|
| Temporal | `anio`, `semestre` | Tendencias históricas |
| Agregados por programa | `prog_mean_rend`, `prog_mean_indicador_*` | Efecto institución/carrera |
| Agregados por departamento | `dept_mean_rend`, `dept_mean_indicador` | Efecto regional |
| Indicadores combinados | `indicador_promedio`, `indicador_std` | Perfil global del estudiante |
| Ratios entre indicadores | `ratio_1_2`, `ratio_1_234` | Balance de competencias |
| Interacciones socioeconómicas | `promedio_x_estrato`, `horas_x_indicadores` | Desigualdad de oportunidades |
| Términos polinómicos | `indicador_1_sq`, `indicador_2_sq` | Relaciones no lineales |

### Modelos Comparados

| Notebook | Modelo | Accuracy (Val.) | Observaciones |
|----------|--------|-----------------|---------------|
| `03` | Random Forest | ~44% | Submuestra 30% por memoria RAM |
| `04` | XGBoost | 44.01% | Pipeline sklearn, CPU |
| `99` | **CatBoost** | — | Solución final, GPU, early stopping |

> **Baseline** (predicción aleatoria): 25%

---

## Estructura del Proyecto

```
udea-kaggle-saberpro/
│
├── 01 - exploración.ipynb                       # EDA completo con estadísticas y visualizaciones
├── 02 - preprocesado.ipynb                      # Limpieza y codificación de variables
├── 03 - modelo con preprocesado y RandomForest.ipynb
├── 04 - modelo con preprocesado y XGBoostClassifier.ipynb
├── 99 - modelo solución.ipynb                   # Solución final con CatBoost
│
├── requirements.txt                             # Dependencias Python
├── .env.example                                 # Plantilla de credenciales Kaggle
├── .gitignore
│
└── data/                                        # Ignorado por git (se genera al ejecutar)
    ├── train.csv                                # Descargado via Kaggle API
    └── train_to_colab.csv                       # Generado por 02 - preprocesado.ipynb
```

---

## Configuración del Entorno

> **Python requerido:** 3.11.9
> Verificar con `python --version` antes de instalar.

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/udea-kaggle-saberpro.git
cd udea-kaggle-saberpro
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Paquetes principales: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `catboost`, `statsmodels`, `python-dotenv`, `kaggle`.

### 3. Configurar credenciales de Kaggle

Copia el archivo de ejemplo y completa tus credenciales:

```bash
cp .env.example .env
```

Edita `.env` con tus datos de [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**:

```env
KAGGLE_USERNAME=tu_usuario_kaggle
KAGGLE_KEY=tu_api_key_de_kaggle
```

> **Alternativa directa:** descarga `kaggle.json` y colócalo en:
> - Linux/Mac: `~/.kaggle/kaggle.json`
> - Windows: `C:\Users\TU_USUARIO\.kaggle\kaggle.json`

### 4. Ejecutar los notebooks en orden

```
01 → 02 → 03 / 04 / 99
```

El notebook `01` descarga automáticamente los datos de Kaggle.
El notebook `02` genera `data/train_to_colab.csv` que usan los modelos.

---

## Videos de Presentación

- **Entrega 2:** https://youtu.be/AZ6GtYL_bX0
- **Entrega Final:** https://youtu.be/Ydbgz5YX_B0

---

## Autor

**Juan David Flórez Obando**
ID: 1017253586 | Ingeniería Industrial | Universidad de Antioquia
