# Smart Onboarding Pipeline 🏦🤖

[![DevSecOps Pipeline](https://github.com/Gusmal02/smart-onboarding-cv/actions/workflows/devsecops_pipeline.yml/badge.svg)](https://github.com/Gusmal02/smart-onboarding-cv/actions)

Pipeline de onboarding inteligente para servicios financieros que combina
**visión artificial** y **modelo de riesgo crediticio** en un solo endpoint REST.

## Problema que resuelve

Los procesos tradicionales de apertura de crédito requieren validación manual
de documentos y evaluación de riesgo por separado, generando tiempos de
onboarding de días y alto costo operativo.

Este pipeline automatiza ambos pasos en una sola llamada API — aplicable
directamente a procesos de crédito en instituciones financieras como Banco Azteca.

## Arquitectura — Sistema Neuro-Simbólico en 3 Capas

Imagen del documento + Datos del cliente

↓

┌─────────────────────────────────────┐

│  CAPA 1 — NEURONAL                  │

│  MobileNetV2 (Transfer Learning)    │

│  Clasificador de documentos         │

│  INE / CURP / Comprobante /         │

│  Estado de cuenta                   │

│  Precisión: 100% (validación)       │

└─────────────────┬───────────────────┘

↓

┌─────────────────────────────────────┐

│  CAPA 2 — PROBABILÍSTICA            │

│  LightGBM — Scoring crediticio      │

│  Bajo / Medio / Alto riesgo         │

│  Accuracy: 94% — F1: 0.90           │

└─────────────────┬───────────────────┘

↓

┌─────────────────────────────────────┐

│  CAPA 3 — SIMBÓLICA                 │

│  Motor de reglas de negocio         │

│  APROBADO / REVISIÓN / RECHAZADO    │

└─────────────────────────────────────┘

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Visión Artificial | PyTorch + MobileNetV2 (Transfer Learning) |
| Modelo de Riesgo | LightGBM |
| API REST | FastAPI + Uvicorn |
| Contenedorización | Docker (usuario no-root) |
| CI/CD + SAST | GitHub Actions + Bandit |
| Gestión de entorno | uv (Astral) |
| Testing | pytest (16 pruebas unitarias) |

## Métricas del Modelo de Riesgo

| Perfil | Precision | Recall | F1 |
|---|---|---|---|
| Bajo Riesgo | 0.99 | 0.97 | 0.98 |
| Riesgo Medio | 0.90 | 0.94 | 0.92 |
| Alto Riesgo | 0.82 | 0.78 | 0.80 |
| **Global** | **0.94** | **0.94** | **0.93** |

## Instalación y ejecución local

```bash
# 1. Clonar repositorio
git clone https://github.com/Gusmal02/smart-onboarding-cv
cd smart-onboarding-cv

# 2. Crear entorno e instalar dependencias
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt

# 3. Generar datos sintéticos
python generate_data.py

# 4. Entrenar modelos
python train.py

# 5. Levantar API
uvicorn api.main:app --reload
```

## Ejecución con Docker

```bash
docker build -t smart-onboarding-cv .
docker run -p 8000:8000 smart-onboarding-cv
```

## Pruebas unitarias

```bash
pytest tests/ -v
# 16 passed
```

## Ejemplo de uso

```json
POST /onboarding

{
  "imagen_base64": "<base64>",
  "edad": 35,
  "ingreso_mensual": 18000,
  "antiguedad_empleo_años": 4.0,
  "num_creditos_activos": 1,
  "dias_mora_historico": 0,
  "monto_solicitado": 20000,
  "tiene_buró_negativo": 0
}

Respuesta:
{
  "tipo_documento": "INE",
  "confianza_documento": 0.952,
  "perfil_riesgo": "Bajo Riesgo",
  "probabilidades_riesgo": {
    "bajo_riesgo": 0.9999,
    "riesgo_medio": 0.0001,
    "alto_riesgo": 0.0
  },
  "decision": "APROBADO",
  "motivo": "Documento válido y perfil de bajo riesgo"
}
```

## Documentación interactiva

Con la API corriendo, accede a:

http://localhost:8000/docs