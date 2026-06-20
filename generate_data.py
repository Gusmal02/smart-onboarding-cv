"""
Generador de datos sintéticos para Smart Onboarding Pipeline
- Imágenes de documentos simuladas con PIL
- Dataset tabular de perfiles de riesgo crediticio
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random

# ── Configuración ──────────────────────────────────────────────────────────────
CLASES = ["INE", "CURP", "COMPROBANTE_DOMICILIO", "ESTADO_CUENTA"]
COLORES = {
    "INE": (180, 30, 30),
    "CURP": (30, 80, 180),
    "COMPROBANTE_DOMICILIO": (30, 140, 60),
    "ESTADO_CUENTA": (120, 60, 180),
}
IMG_DIR = "data/synthetic_images"
CSV_PATH = "data/synthetic_clients.csv"
IMGS_POR_CLASE = 80  # 320 imágenes total — suficiente para demo
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Generador de imágenes ──────────────────────────────────────────────────────
def generar_imagen_documento(clase: str, idx: int) -> Image.Image:
    """Genera imagen sintética que simula un documento oficial."""
    w, h = 400, 250
    img = Image.new("RGB", (w, h), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)

    color = COLORES[clase]

    # Borde del documento
    draw.rectangle([10, 10, w - 10, h - 10], outline=color, width=4)

    # Banda de color superior (header)
    draw.rectangle([10, 10, w - 10, 60], fill=color)

    # Texto del tipo de documento
    draw.text((20, 20), clase.replace("_", " "), fill=(255, 255, 255))
    draw.text((20, 40), "GOBIERNO DE MÉXICO", fill=(220, 220, 220))

    # Simulación de campos
    campos = ["NOMBRE:", "FECHA:", "FOLIO:", "DIRECCIÓN:"]
    for i, campo in enumerate(campos):
        y = 75 + i * 35
        draw.text((25, y), campo, fill=(80, 80, 80))
        # Línea simulando dato
        draw.line([120, y + 10, 350, y + 10], fill=(180, 180, 180), width=2)

    # Ruido visual para variación entre imágenes
    noise_level = random.randint(0, 30)
    pixels = img.load()
    for _ in range(noise_level * 10):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        pixels[x, y] = (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255),
        )

    # Leve rotación para simular escaneos reales
    angulo = random.uniform(-3, 3)
    img = img.rotate(angulo, fillcolor=(245, 245, 240))

    return img


def generar_dataset_imagenes():
    """Genera y guarda todas las imágenes por clase."""
    print("Generando imágenes sintéticas...")
    for clase in CLASES:
        clase_dir = os.path.join(IMG_DIR, clase)
        os.makedirs(clase_dir, exist_ok=True)
        for i in range(IMGS_POR_CLASE):
            img = generar_imagen_documento(clase, i)
            img.save(os.path.join(clase_dir, f"{clase}_{i:03d}.png"))
    print(f"✅ {len(CLASES) * IMGS_POR_CLASE} imágenes generadas en {IMG_DIR}/")


# ── Generador de perfiles de riesgo ───────────────────────────────────────────
def generar_dataset_clientes(n: int = 1000) -> pd.DataFrame:
    """
    Genera dataset tabular de clientes con perfil de riesgo crediticio.
    Features basadas en variables reales de scoring financiero.
    """
    data = {
        "edad": np.random.randint(18, 70, n),
        "ingreso_mensual": np.random.exponential(scale=8000, size=n).clip(2000, 80000),
        "antiguedad_empleo_años": np.random.exponential(scale=3, size=n).clip(0, 30),
        "num_creditos_activos": np.random.randint(0, 8, n),
        "dias_mora_historico": np.random.exponential(scale=15, size=n).clip(0, 180),
        "tipo_documento": np.random.choice(
            [0, 1, 2, 3], n, p=[0.6, 0.2, 0.1, 0.1]
        ),  # 0=INE, 1=CURP, 2=Comp, 3=Estado
        "monto_solicitado": np.random.exponential(scale=15000, size=n).clip(
            1000, 150000
        ),
        "tiene_buró_negativo": np.random.choice([0, 1], n, p=[0.75, 0.25]),
    }

    df = pd.DataFrame(data)

    # Lógica de riesgo basada en reglas de negocio reales
    def calcular_riesgo(row):
        score = 0
        if row["ingreso_mensual"] < 5000:
            score += 2
        if row["dias_mora_historico"] > 30:
            score += 3
        if row["tiene_buró_negativo"] == 1:
            score += 2
        if row["num_creditos_activos"] > 4:
            score += 1
        if row["antiguedad_empleo_años"] < 1:
            score += 1
        if row["monto_solicitado"] / max(row["ingreso_mensual"], 1) > 5:
            score += 2

        if score <= 2:
            return 0  # Bajo riesgo
        elif score <= 5:
            return 1  # Riesgo medio
        else:
            return 2  # Alto riesgo

    df["perfil_riesgo"] = df.apply(calcular_riesgo, axis=1)

    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    dist = df["perfil_riesgo"].value_counts().sort_index()
    print(f"✅ Dataset de {n} clientes generado en {CSV_PATH}")
    print(f"   Bajo riesgo:  {dist.get(0, 0)} clientes")
    print(f"   Riesgo medio: {dist.get(1, 0)} clientes")
    print(f"   Alto riesgo:  {dist.get(2, 0)} clientes")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Smart Onboarding Pipeline — Generador de Datos")
    print("=" * 50)
    generar_dataset_imagenes()
    generar_dataset_clientes()
    print("\n✅ Datos listos. Ejecuta train.py para entrenar los modelos.")