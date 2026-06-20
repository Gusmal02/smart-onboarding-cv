"""
Suite de pruebas unitarias — Smart Onboarding Pipeline
Valida integridad del pipeline sin depender de modelos entrenados
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Tests de generación de datos ───────────────────────────────────────────────
class TestGeneracionDatos:

    def test_dataset_clientes_existe(self):
        """Verifica que el CSV de clientes fue generado."""
        assert os.path.exists("data/synthetic_clients.csv"), \
            "CSV de clientes no encontrado"

    def test_dataset_clientes_columnas(self):
        """Verifica columnas requeridas en el dataset."""
        df = pd.read_csv("data/synthetic_clients.csv")
        columnas_requeridas = [
            "edad", "ingreso_mensual", "antiguedad_empleo_años",
            "num_creditos_activos", "dias_mora_historico",
            "tipo_documento", "monto_solicitado",
            "tiene_buró_negativo", "perfil_riesgo"
        ]
        for col in columnas_requeridas:
            assert col in df.columns, f"Columna faltante: {col}"

    def test_dataset_clientes_sin_nulos(self):
        """Verifica que no hay valores nulos."""
        df = pd.read_csv("data/synthetic_clients.csv")
        assert df.isnull().sum().sum() == 0, "Dataset contiene valores nulos"

    def test_dataset_clientes_riesgo_valido(self):
        """Verifica que perfil_riesgo solo tiene valores 0, 1, 2."""
        df = pd.read_csv("data/synthetic_clients.csv")
        valores_validos = {0, 1, 2}
        assert set(df["perfil_riesgo"].unique()).issubset(valores_validos)

    def test_imagenes_generadas(self):
        """Verifica que se generaron imágenes para cada clase."""
        clases = ["INE", "CURP", "COMPROBANTE_DOMICILIO", "ESTADO_CUENTA"]
        for clase in clases:
            ruta = f"data/synthetic_images/{clase}"
            assert os.path.exists(ruta), f"Carpeta faltante: {ruta}"
            imagenes = [f for f in os.listdir(ruta) if f.endswith(".png")]
            assert len(imagenes) == 80, \
                f"Se esperaban 80 imágenes en {clase}, hay {len(imagenes)}"

    def test_imagenes_dimensiones(self):
        """Verifica que las imágenes tienen dimensiones correctas."""
        img = Image.open("data/synthetic_images/INE/INE_000.png")
        assert img.size == (400, 250), f"Dimensión incorrecta: {img.size}"
        assert img.mode == "RGB", f"Modo incorrecto: {img.mode}"


# ── Tests de modelos ───────────────────────────────────────────────────────────
class TestModelos:

    def test_modelo_doc_existe(self):
        """Verifica que el modelo de documentos fue guardado."""
        assert os.path.exists("models/doc_classifier.pth"), \
            "Modelo clasificador no encontrado"

    def test_modelo_riesgo_existe(self):
        """Verifica que el modelo de riesgo fue guardado."""
        assert os.path.exists("models/risk_model.pkl"), \
            "Modelo de riesgo no encontrado"

    def test_modelo_riesgo_prediccion(self):
        """Verifica que el modelo de riesgo predice correctamente."""
        import pickle
        with open("models/risk_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Cliente de bajo riesgo obvio
        X = np.array([[35, 20000, 5, 1, 0, 0, 15000, 0]])
        pred = model.predict(X)
        assert pred[0] in [0, 1, 2], f"Predicción fuera de rango: {pred[0]}"

    def test_modelo_riesgo_probabilidades(self):
        """Verifica que las probabilidades suman 1."""
        import pickle
        with open("models/risk_model.pkl", "rb") as f:
            model = pickle.load(f)

        X = np.array([[35, 20000, 5, 1, 0, 0, 15000, 0]])
        probs = model.predict_proba(X)[0]
        assert abs(probs.sum() - 1.0) < 1e-6, \
            f"Probabilidades no suman 1: {probs.sum()}"


# ── Tests de lógica de negocio ─────────────────────────────────────────────────
class TestLogicaNegocio:

    def setup_method(self):
        """Importa la función de decisión."""
        sys.path.insert(0, "api")
        from api.main import tomar_decision
        self.tomar_decision = tomar_decision

    def test_decision_aprobado(self):
        """Cliente ideal debe ser aprobado."""
        decision, _ = self.tomar_decision("INE", "Bajo Riesgo", 0.95)
        assert decision == "APROBADO"

    def test_decision_rechazado_alto_riesgo(self):
        """Alto riesgo debe ser rechazado."""
        decision, _ = self.tomar_decision("INE", "Alto Riesgo", 0.95)
        assert decision == "RECHAZADO"

    def test_decision_revision_baja_confianza(self):
        """Baja confianza en documento debe ir a revisión."""
        decision, _ = self.tomar_decision("INE", "Bajo Riesgo", 0.50)
        assert decision == "REVISIÓN MANUAL"

    def test_decision_revision_documento_invalido(self):
        """Documento que no es INE/CURP debe ir a revisión."""
        decision, _ = self.tomar_decision("ESTADO_CUENTA", "Bajo Riesgo", 0.95)
        assert decision == "REVISIÓN MANUAL"

    def test_decision_revision_riesgo_medio(self):
        """Riesgo medio siempre va a revisión."""
        decision, _ = self.tomar_decision("INE", "Riesgo Medio", 0.95)
        assert decision == "REVISIÓN MANUAL"

    def test_sin_division_por_cero(self):
        """Ingreso cero no debe causar excepción en el modelo."""
        import pickle
        with open("models/risk_model.pkl", "rb") as f:
            model = pickle.load(f)
        X = np.array([[25, 0.0, 0, 0, 0, 0, 1000, 0]])
        try:
            pred = model.predict(X)
            assert pred[0] in [0, 1, 2]
        except ZeroDivisionError:
            pytest.fail("División por cero con ingreso_mensual=0")