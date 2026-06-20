"""
Smart Onboarding Pipeline — API REST
Endpoint único que recibe imagen + datos del cliente
y devuelve clasificación de documento + perfil de riesgo
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# ── Configuración ──────────────────────────────────────────────────────────────
CLASES_DOC = ["INE", "CURP", "COMPROBANTE_DOMICILIO", "ESTADO_CUENTA"]
CLASES_RIESGO = ["Bajo Riesgo", "Riesgo Medio", "Alto Riesgo"]
MODELOS_DIR = "models"

app = FastAPI(
    title="Smart Onboarding Pipeline",
    description="""
    Pipeline de onboarding inteligente para servicios financieros.
    Combina visión artificial para clasificación de documentos
    con modelo de riesgo crediticio basado en LightGBM.
    
    Desarrollado como prueba de concepto aplicable a procesos
    de apertura de crédito en instituciones financieras.
    """,
    version="1.0.0",
)


# ── Carga de modelos ───────────────────────────────────────────────────────────
def cargar_clasificador():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 4),
    )
    model.load_state_dict(
        torch.load(
            os.path.join(MODELOS_DIR, "doc_classifier.pth"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    return model, device


def cargar_modelo_riesgo():
    with open(os.path.join(MODELOS_DIR, "risk_model.pkl"), "rb") as f:
        return pickle.load(f)  # nosec B301


# Carga al iniciar
print("Cargando modelos...")
clf_doc, device = cargar_clasificador()
clf_riesgo = cargar_modelo_riesgo()
print("✅ Modelos listos")

# Transform para inferencia
transform_inferencia = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Schemas ────────────────────────────────────────────────────────────────────
class ClienteRequest(BaseModel):
    imagen_base64: str
    edad: int
    ingreso_mensual: float
    antiguedad_empleo_años: float
    num_creditos_activos: int
    dias_mora_historico: float
    monto_solicitado: float
    tiene_buró_negativo: int

    class Config:
        json_schema_extra = {
            "example": {
                "imagen_base64": "<base64 de imagen de documento>",
                "edad": 35,
                "ingreso_mensual": 15000.0,
                "antiguedad_empleo_años": 3.5,
                "num_creditos_activos": 2,
                "dias_mora_historico": 0.0,
                "monto_solicitado": 25000.0,
                "tiene_buró_negativo": 0,
            }
        }


class OnboardingResponse(BaseModel):
    tipo_documento: str
    confianza_documento: float
    perfil_riesgo: str
    probabilidades_riesgo: dict
    decision: str
    motivo: str


# ── Motor de decisión ──────────────────────────────────────────────────────────
def tomar_decision(tipo_doc: str, perfil_riesgo: str, confianza: float) -> tuple:
    """
    Capa simbólica: reglas de negocio sobre las predicciones del modelo.
    Inspirado en arquitectura neuro-simbólica.
    """
    if confianza < 0.7:
        return "REVISIÓN MANUAL", "Documento con baja confianza de clasificación"

    if tipo_doc not in ["INE", "CURP"]:
        return "REVISIÓN MANUAL", "Se requiere INE o CURP como documento principal"

    if perfil_riesgo == "Alto Riesgo":
        return "RECHAZADO", "Perfil de alto riesgo crediticio detectado"

    if perfil_riesgo == "Riesgo Medio":
        return "REVISIÓN MANUAL", "Perfil de riesgo medio requiere validación adicional"

    return "APROBADO", "Documento válido y perfil de bajo riesgo"


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "proyecto": "Smart Onboarding Pipeline",
        "version": "1.0.0",
        "modelos": ["MobileNetV2 (Transfer Learning)", "LightGBM"],
        "endpoint_principal": "/onboarding",
        "documentacion": "/docs",
    }


@app.post("/onboarding", response_model=OnboardingResponse)
def onboarding(cliente: ClienteRequest):
    """
    Endpoint principal del pipeline.
    Recibe imagen del documento + datos del cliente.
    Devuelve clasificación, perfil de riesgo y decisión automatizada.
    """
    # ── Clasificación de documento ─────────────────────────────────────────
    try:
        img_bytes = base64.b64decode(cliente.imagen_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen base64 inválida")

    tensor = transform_inferencia(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = clf_doc(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx_doc = probs.argmax().item()
        confianza_doc = probs[idx_doc].item()
        tipo_documento = CLASES_DOC[idx_doc]

    # ── Perfil de riesgo ───────────────────────────────────────────────────
    features = np.array([[
        cliente.edad,
        cliente.ingreso_mensual,
        cliente.antiguedad_empleo_años,
        cliente.num_creditos_activos,
        cliente.dias_mora_historico,
        CLASES_DOC.index(tipo_documento),  # tipo doc inferido por visión
        cliente.monto_solicitado,
        cliente.tiene_buró_negativo,
    ]])

    pred_riesgo = clf_riesgo.predict(features)[0]
    probs_riesgo = clf_riesgo.predict_proba(features)[0]
    perfil_riesgo = CLASES_RIESGO[pred_riesgo]

    # ── Decisión neuro-simbólica ───────────────────────────────────────────
    decision, motivo = tomar_decision(tipo_documento, perfil_riesgo, confianza_doc)

    return OnboardingResponse(
        tipo_documento=tipo_documento,
        confianza_documento=round(confianza_doc, 4),
        perfil_riesgo=perfil_riesgo,
        probabilidades_riesgo={
            "bajo_riesgo": round(float(probs_riesgo[0]), 4),
            "riesgo_medio": round(float(probs_riesgo[1]), 4),
            "alto_riesgo": round(float(probs_riesgo[2]), 4),
        },
        decision=decision,
        motivo=motivo,
    )


@app.get("/health")
def health():
    return {"status": "ok", "modelos_cargados": True}
