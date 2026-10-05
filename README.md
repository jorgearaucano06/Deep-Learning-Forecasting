# Mantenimiento Predictivo de Redes de Fibra Optica con Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.13+-ff6f00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-2.5+-0194E2?logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-En%20desarrollo-yellow" />
</p>

---

## Problema de Negocio

Las redes de fibra optica GPON son la columna vertebral de las telecomunicaciones. Cuando un segmento falla sin aviso, el impacto es inmediato:

| Impacto | Mantenimiento Reactivo | Mantenimiento Predictivo |
|---------|----------------------|------------------------|
| Deteccion de falla | Despues del evento | **24h antes del evento** |
| Downtime promedio | 4-12 horas | **~0 horas** |
| Costo por falla | ~$3,000 USD | ~$150 USD (inspeccion) |
| Tipo de reparacion | Emergencia (nocturna/fds) | Planificada (horario normal) |
| SLA compliance | 99.2% | **99.95%** |

**Este sistema predice fallas en segmentos de fibra optica con 24 horas de anticipacion**, clasificando el tipo de falla y priorizando la respuesta del equipo tecnico.

**Ahorro estimado: 86% en costos de mantenimiento.**

---

## Arquitectura del Sistema

```
                          PIPELINE DE DATOS
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Sensores GPON        Feature Engineering         Split Temporal    │
│  (cada 15 min)        (80+ features)              (Train/Val/Test)  │
│                                                                     │
│  ┌──────────┐    ┌─────────────────────┐    ┌──────────────────┐   │
│  │ Potencia │    │ Rolling stats (3h)  │    │ Train: Dias 1-5  │   │
│  │ OSNR     │───>│ Lags (15min-12h)    │───>│ Val:   Dia  5-6  │   │
│  │ BER      │    │ Tasas de cambio     │    │ Test:  Dias 6-7  │   │
│  │ Atenua.  │    │ Health score        │    │ (orden temporal)  │   │
│  │ Disp. CD │    │ Codif. ciclica hora │    └────────┬─────────┘   │
│  └──────────┘    └─────────────────────┘             │              │
│                                                       │              │
└───────────────────────────────────────────────────────┼──────────────┘
                                                        │
                          MODELOS                       │
┌───────────────────────────────────────────────────────┼──────────────┐
│                                                       v              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │  Autoencoder│  │   CNN 1D     │  │   BiLSTM + Attention       │  │
│  │  (Anomalia) │  │ (Clasifica)  │  │   (Prediccion temporal)    │  │
│  │             │  │              │  │                            │  │
│  │  ~8K params │  │  ~50K params │  │  ~400K params              │  │
│  │  No superv. │  │  4 clases    │  │  Ventana 12h -> pred 24h  │  │
│  └──────┬──────┘  └──────┬───────┘  └────────────┬───────────────┘  │
│         │                │                        │                  │
│         v                v                        v                  │
│    "Hay algo         "Es corte,              "Falla en             │
│     raro"            degradacion             ~18 horas"             │
│                      o empalme"                                     │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                          DEPLOY
┌─────────────────────────────┼────────────────────────────────────────┐
│                             v                                        │
│         ┌──────────────────────────────────┐                        │
│         │         FastAPI REST API          │                        │
│         │  POST /predict/{segment_id}       │                        │
│         │  POST /predict/batch              │                        │
│         └──────────────┬───────────────────┘                        │
│                        │                                             │
│         ┌──────────────v───────────────────┐                        │
│         │       Plotly Dash Dashboard       │                        │
│         │  Mapa de riesgo | KPIs | Timeline │                        │
│         └──────────────────────────────────┘                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Metricas Monitoreadas

Los valores estan calibrados con estandares internacionales de la industria:

| Metrica | Rango Normal | Estandar de Referencia | Descripcion |
|---------|-------------|----------------------|-------------|
| Potencia Optica | -18 a -8 dBm | ITU-T G.984.2 (GPON) | Intensidad de senal recibida |
| Atenuacion | 0.18-0.25 dB/km | ITU-T G.652 (G.652.D) | Perdida de senal por kilometro |
| BER | 1e-12 a 1e-9 | ITU-T G.984.2 | Tasa de errores de bit |
| OSNR | 20-35 dB | IEEE / OPM Papers | Relacion senal optica / ruido |
| Dispersion Cromatica | 15-18 ps/(nm*km) | ITU-T G.652 @1550nm | Ensanchamiento del pulso optico |

### Tipos de Falla Detectados

| Tipo | Proporcion | Firma en Sensores | Causa Tipica |
|------|-----------|-------------------|--------------|
| Corte fisico | 25% | Caida abrupta de potencia a -40 dBm, BER > 1e-4 | Excavaciones, roedores, vandalismo |
| Degradacion gradual | 50% | Deterioro lento en potencia, BER y atenuacion | Envejecimiento, humedad, estres mecanico |
| Empalme defectuoso | 25% | Oscilaciones periodicas de potencia (> 0.5 dB) | Fusion incompleta, desalineacion |

---

## Modelos Implementados

### Comparacion de Modelos

| Modelo | Framework | Tarea | Input Shape | Params | Proposito |
|--------|-----------|-------|-------------|--------|-----------|
| **BiLSTM + Attention** | PyTorch | Prediccion de falla 24h | (batch, 48, 12) | ~400K | Modelo principal - captura tendencias temporales |
| **CNN 1D** | PyTorch | Clasificacion 4 clases | (batch, 12, 48) | ~50K | Clasificacion rapida del tipo de falla |
| **Autoencoder** | PyTorch | Deteccion de anomalias | (batch, 12) | ~8K | Detecta fallas no vistas (no supervisado) |
| **LSTM** | TensorFlow | Prediccion de falla 24h | (batch, 48, 12) | ~350K | Implementacion dual-framework |
| **Random Forest** | scikit-learn | Clasificacion + baseline | (n, features) | N/A | Baseline + feature importance |
| **ARIMA** | statsmodels | Prediccion univariada | serie temporal | N/A | Baseline estadistico |

### Modelo Principal: BiLSTM con Mecanismo de Atencion

```
Input: (batch, 48 timestamps, 12 features)
  │       12 horas de lecturas cada 15 min
  │       con 12 metricas por lectura
  v
┌─────────────────────────────────────┐
│  BiLSTM Capa 1 (128 x 2 dirs)      │  Lee la secuencia en ambas direcciones
│  + Dropout 0.3                      │  Output: (batch, 48, 256)
├─────────────────────────────────────┤
│  BiLSTM Capa 2 (128 x 2 dirs)      │  Captura patrones mas abstractos
│  + Dropout 0.3                      │  Output: (batch, 48, 256)
├─────────────────────────────────────┤
│  Attention Layer                    │  Pesos por timestamp: enfoca en
│                                     │  momentos criticos de la secuencia
│                                     │  Output: (batch, 256)
├─────────────────────────────────────┤
│  Dense 256 → 64 + ReLU + Dropout   │
│  Dense 64 → 1 + Sigmoid            │  Output: probabilidad [0, 1]
└─────────────────────────────────────┘
  │
  v
P(falla en 24h) = 0.87 → ALERTA
```

### Uso Complementario en Produccion

```
Paso 1: Autoencoder  →  "Segmento SEG-047: anomalia detectada (score: 3.8)"
Paso 2: CNN 1D       →  "Clasificacion: degradacion gradual (83% confianza)"
Paso 3: BiLSTM       →  "Falla estimada en ~18 horas"
Paso 4: Dashboard    →  "Prioridad: MEDIA - programar tecnico turno siguiente"
```

---

## Datos

### Generacion de Datos Sinteticos Calibrados

Los datos sinteticos replican el comportamiento real de una red GPON, calibrados con estandares de la industria:

- **Red simulada**: 200 segmentos de fibra, 7 dias, lecturas cada 15 minutos (~134,000 registros)
- **Propiedades por segmento**: Longitud (5-20 km), empalmes, conectores, potencia base calculada con **link budget real**
- **Link budget**: Potencia OLT - perdida fibra - perdida empalmes - perdida conectores - perdida splitter 1:32
- **Firmas de falla**: Inyectadas 12-72h antes del evento, basadas en patrones documentados en trazas OTDR (IEEE DataPort)
- **Ruido de medicion**: Basado en precision de equipos OTDR comerciales (EXFO, VIAVI)

| Parametro | Valor | Fuente |
|-----------|-------|--------|
| Tipo de fibra | G.652.D monomodo | ITU-T G.652 |
| Longitud de onda | 1550 nm | ITU-T G.984.2 |
| Red | GPON, split ratio 1:32 | ITU-T G.984.x |
| Perdida por empalme | 0.05-0.15 dB (bueno) | FOA (Fiber Optic Assoc.) |
| Perdida por conector | 0.3-0.5 dB | TIA/EIA-568 |
| Perdida splitter 1:32 | 17.5 dB | Especificacion GPON |

### Feature Engineering

De 5 metricas base se generan **80+ features derivadas**:

| Categoria | Ejemplos | Cantidad |
|-----------|----------|----------|
| Rolling statistics | Media, std, min, max en ventanas de 1h, 3h, 12h | ~40 |
| Lag features | Valores 15min, 1h, 3h, 12h atras | ~20 |
| Tasas de cambio | Diferencias y ratios entre timestamps | ~10 |
| Features de dominio | Health score (0-100), power/attenuation ratio | ~5 |
| Temporales | Hora del dia (codificacion ciclica sin/cos) | ~5 |

---

## Estructura del Proyecto

```
Deep-Learning-Forecasting/
│
├── config/
│   └── config.yaml                  # Hiperparametros, rangos, constantes (con fuentes ITU-T)
│
├── data/
│   ├── raw/                         # Datos generados (parquet) - generado por scripts
│   └── processed/                   # Features procesadas (parquet) - generado por pipeline
│
├── notebooks/                       # Ejecucion secuencial 01 → 09
│   ├── 01_data_generation.ipynb     # Generacion de datos sinteticos
│   ├── 02_eda.ipynb                 # Analisis exploratorio + visualizaciones
│   ├── 03_feature_engineering.ipynb # Pipeline de features
│   ├── 04_baseline_models.ipynb     # ARIMA + Random Forest
│   ├── 05_lstm_pytorch.ipynb        # BiLSTM + Attention (PyTorch)
│   ├── 06_autoencoder_anomaly.ipynb # Autoencoder deteccion anomalias
│   ├── 07_cnn1d_classification.ipynb# CNN 1D clasificacion 4 clases
│   ├── 08_tensorflow_model.ipynb    # LSTM (TensorFlow/Keras)
│   └── 09_model_comparison.ipynb    # Comparacion de modelos + KPIs de negocio
│
├── src/
│   ├── data/
│   │   ├── generator.py             # Generador de datos sinteticos (link budget real)
│   │   ├── preprocessor.py          # Limpieza, normalizacion, split temporal
│   │   ├── feature_engineer.py      # Pipeline de feature engineering
│   │   └── dataset.py               # PyTorch Datasets + DataLoaders
│   ├── models/
│   │   ├── lstm_predictor.py        # BiLSTM + Attention (PyTorch)
│   │   ├── autoencoder.py           # Autoencoder deteccion anomalias
│   │   ├── cnn1d_classifier.py      # CNN 1D clasificacion de fallas
│   │   ├── tf_lstm_predictor.py     # LSTM (TensorFlow/Keras)
│   │   └── baseline.py              # ARIMA + Random Forest
│   ├── training/
│   │   ├── trainer.py               # Training loop generico + early stopping
│   │   ├── evaluator.py             # Metricas tecnicas + comparacion
│   │   └── experiment.py            # MLflow experiment tracking
│   ├── api/
│   │   ├── main.py                  # FastAPI application
│   │   ├── routes.py                # Endpoints REST
│   │   ├── schemas.py               # Pydantic request/response models
│   │   └── model_service.py         # Servicio de carga e inferencia
│   ├── dashboard/
│   │   ├── app.py                   # Plotly Dash application
│   │   └── components.py            # Componentes visuales (mapa, KPIs, timeline)
│   └── utils/
│       ├── config_loader.py         # Carga de configuracion YAML
│       ├── logger.py                # Logging centralizado
│       ├── metrics.py               # Metricas de negocio (ROI, ahorro, downtime)
│       └── visualization.py         # Funciones de ploteo reutilizables
│
├── scripts/
│   ├── generate_data.py             # CLI: generar datos sinteticos
│   ├── train_all_models.py          # CLI: entrenar todos los modelos
│   ├── run_api.py                   # CLI: levantar API REST
│   └── run_dashboard.py             # CLI: levantar dashboard
│
├── models/                          # Modelos entrenados (.pt, .h5) - generado por training
├── mlruns/                          # MLflow tracking - generado por experiments
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalacion y Ejecucion

### Requisitos previos
- Python 3.9+
- pip

### Setup

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/Deep-Learning-Forecasting.git
cd Deep-Learning-Forecasting

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecucion

```bash
# 1. Generar datos sinteticos (~134,000 registros)
python scripts/generate_data.py

# 2. Entrenar todos los modelos (con MLflow tracking)
python scripts/train_all_models.py

# 3. Levantar API REST
python scripts/run_api.py
# → http://localhost:8000/docs (Swagger UI)

# 4. Levantar Dashboard
python scripts/run_dashboard.py
# → http://localhost:8050
```

### Notebooks
Los notebooks estan disenados para ejecucion secuencial (`01` → `09`).
Cada notebook es independiente pero sigue el flujo logico del pipeline.

---

## API REST

### Endpoints

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Estado del servicio y modelos cargados |
| `POST` | `/api/v1/predict/{segment_id}` | Prediccion para un segmento |
| `POST` | `/api/v1/predict/batch` | Prediccion en lote (multiples segmentos) |

### Ejemplo de Request

```bash
curl -X POST http://localhost:8000/api/v1/predict/SEG-001 \
  -H "Content-Type: application/json" \
  -d '[{
    "optical_power_dbm": -14.5,
    "attenuation_db_km": 0.22,
    "ber": 1e-10,
    "osnr_db": 28.0,
    "chromatic_dispersion": 16.5
  }]'
```

### Ejemplo de Response

```json
{
  "segment_id": "SEG-001",
  "risk_level": "medium",
  "failure_probability": 0.34,
  "predicted_class": "degradation",
  "confidence": 0.78,
  "recommended_action": "Programar inspeccion en proximas 48h",
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## Impacto de Negocio

### KPIs Proyectados

| KPI | Sin Sistema | Con Sistema | Mejora |
|-----|-----------|-----------|--------|
| Deteccion temprana (24h+) | 0% | ~88% | — |
| Costo promedio por falla | ~$3,000 | ~$150 | **-95%** |
| Fallas no detectadas | 100% | ~12% | **-88%** |
| Downtime mensual | ~12 horas | ~1 hora | **-92%** |
| SLA compliance | 99.2% | 99.95% | **+0.75pp** |
| Ahorro anual estimado (200 seg.) | — | ~$250,000 | — |

### Flujo de Valor

```
Deteccion temprana     →  Tecnico llega preparado  →  Sin downtime
(24h anticipacion)        (sabe tipo de falla)        (servicio continuo)
       │                         │                          │
       v                         v                          v
  Menos emergencias        Menos costo/visita        Clientes satisfechos
  (88% prevenidas)         ($150 vs $3,000)          (SLA 99.95%)
```

---

## Decisiones Tecnicas

| Decision | Justificacion |
|----------|--------------|
| **Datos sinteticos calibrados** | Permiten desarrollar el pipeline completo sin depender de datos propietarios. Todos los rangos estan basados en estandares ITU-T, FOA y TIA/EIA. El pipeline es identico al que se usaria con datos reales del NMS. |
| **Split temporal (no aleatorio)** | En series temporales, mezclar pasado y futuro causa data leakage. El split cronologico simula el uso real: entrenar con el pasado, predecir el futuro. |
| **6 modelos complementarios** | Cada modelo tiene fortalezas distintas. Random Forest da interpretabilidad (feature importance), el LSTM captura tendencias temporales, el Autoencoder detecta fallas no vistas, la CNN 1D clasifica rapido en produccion. |
| **PyTorch + TensorFlow** | Implementacion en ambos frameworks para demostrar competencia dual. PyTorch como principal (control fino del training loop), TensorFlow como alternativa (Keras para prototipado rapido). |
| **Metricas de negocio** | Precision y recall no significan nada para un gerente. Traducir a dolares ahorrados, horas de downtime prevenido y SLA compliance comunica valor real. |
| **Recall > Precision** | En mantenimiento predictivo, una falla no detectada (FN) cuesta mucho mas que una falsa alarma (FP). El sistema esta calibrado para preferir falsas alarmas. |

---

## Stack Tecnologico

| Categoria | Tecnologias |
|-----------|------------|
| Deep Learning | PyTorch 2.0+, TensorFlow 2.13+ / Keras |
| ML Clasico | scikit-learn, statsmodels (ARIMA) |
| Datos | pandas, NumPy, PyArrow (Parquet) |
| Visualizacion | Matplotlib, Seaborn, Plotly |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Plotly Dash |
| Experiment Tracking | MLflow |
| Configuracion | YAML (pyyaml) |

---

## Contexto de Dominio

### Red GPON (Gigabit Passive Optical Network)

```
                    ┌──── ONT-001 (Cliente)
                    │
OLT ────── Fibra ───┤──── ONT-002 (Cliente)
(Central)  (G.652.D)│
                    ├──── ONT-003 (Cliente)
            Splitter│
             1:32   ├──── ...
                    │
                    └──── ONT-032 (Cliente)
```

- **OLT** (Optical Line Terminal): Equipo central que transmite a todos los clientes
- **ONT** (Optical Network Terminal): Equipo en la casa del cliente
- **Splitter 1:32**: Divide la senal optica entre 32 clientes (perdida de ~17.5 dB)
- **Fibra G.652.D**: Fibra monomodo estandar, 0.20 dB/km de atenuacion tipica a 1550nm

### Equipos de Monitoreo en la Vida Real

| Equipo | Funcion | Dato que Genera | Formato |
|--------|---------|-----------------|---------|
| OTDR (EXFO, VIAVI) | Pulso de luz que mide distancia y reflexion | Traza de atenuacion por metro | Archivos .SOR |
| OLT (Huawei, ZTE) | Monitoreo continuo de la red | Potencia, BER, estado ONTs | SNMP/OMCI → NMS |
| ONT | Monitoreo en punto del cliente | Potencia recibida, temperatura | SNMP traps |
| NMS | Recopila todo en base de datos central | Series temporales consolidadas | SQL / CSV / API |

---

## Licencia

MIT License
