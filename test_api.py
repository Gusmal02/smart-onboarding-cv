"""
Prueba del endpoint /onboarding con imagen sintética real del dataset
"""
import base64
import json
import urllib.request
import urllib.error

# Carga una imagen real del dataset generado
IMG_PATH = "data/synthetic_images/INE/INE_000.png"

with open(IMG_PATH, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

# Payload de prueba — cliente de bajo riesgo
payload_bajo = {
    "imagen_base64": img_b64,
    "edad": 35,
    "ingreso_mensual": 18000.0,
    "antiguedad_empleo_años": 4.0,
    "num_creditos_activos": 1,
    "dias_mora_historico": 0.0,
    "monto_solicitado": 20000.0,
    "tiene_buró_negativo": 0,
}

# Payload de prueba — cliente de alto riesgo
payload_alto = {
    "imagen_base64": img_b64,
    "edad": 22,
    "ingreso_mensual": 3500.0,
    "antiguedad_empleo_años": 0.3,
    "num_creditos_activos": 6,
    "dias_mora_historico": 90.0,
    "monto_solicitado": 80000.0,
    "tiene_buró_negativo": 1,
}

def llamar_api(payload, etiqueta):
    print(f"\n{'='*50}")
    print(f"CASO: {etiqueta}")
    print('='*50)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/onboarding",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            print(f"Documento detectado : {result['tipo_documento']}")
            print(f"Confianza           : {result['confianza_documento']*100:.1f}%")
            print(f"Perfil de riesgo    : {result['perfil_riesgo']}")
            print(f"Probabilidades      : {result['probabilidades_riesgo']}")
            print(f"DECISIÓN            : {result['decision']}")
            print(f"Motivo              : {result['motivo']}")
    except urllib.error.HTTPError as e:
        print(f"Error {e.code}: {e.read().decode()}")

llamar_api(payload_bajo, "Cliente perfil BAJO RIESGO")
llamar_api(payload_alto, "Cliente perfil ALTO RIESGO")

print("\n✅ Prueba completada")