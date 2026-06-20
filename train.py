"""
Smart Onboarding Pipeline — Entrenamiento
Módulo 1: Clasificador de documentos (MobileNetV2 + Transfer Learning)
Módulo 2: Modelo de riesgo crediticio (LightGBM)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASES = ["INE", "CURP", "COMPROBANTE_DOMICILIO", "ESTADO_CUENTA"]
IMG_DIR = "data/synthetic_images"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Dataset personalizado ──────────────────────────────────────────────────────
class DocumentDataset(Dataset):
    def __init__(self, img_dir, clases, transform=None):
        self.samples = []
        self.transform = transform
        for idx, clase in enumerate(clases):
            clase_dir = os.path.join(img_dir, clase)
            for fname in os.listdir(clase_dir):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(clase_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Módulo 1: Clasificador de documentos ──────────────────────────────────────
def entrenar_clasificador_documentos():
    print("\n" + "=" * 50)
    print("MÓDULO 1: Clasificador de Documentos (MobileNetV2)")
    print("=" * 50)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = DocumentDataset(IMG_DIR, CLASES, transform=transform_train)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = transform_val

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    # Transfer Learning — MobileNetV2 pre-entrenado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Congelar capas base — solo entrenamos el clasificador final
    for param in model.features.parameters():
        param.requires_grad = False

    # Reemplazar clasificador para 4 clases
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 4),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    # Entrenamiento — 10 épocas suficientes para demo
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validación
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        print(f"Época {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {running_loss/len(train_loader):.3f} | "
              f"Train Acc: {100*correct/total:.1f}% | "
              f"Val Acc: {100*val_correct/val_total:.1f}%")

    # Guardar modelo
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "doc_classifier.pth"))
    print(f"✅ Modelo guardado en {MODELS_DIR}/doc_classifier.pth")
    return model


# ── Módulo 2: Modelo de riesgo crediticio ─────────────────────────────────────
def entrenar_modelo_riesgo():
    print("\n" + "=" * 50)
    print("MÓDULO 2: Modelo de Riesgo Crediticio (LightGBM)")
    print("=" * 50)

    df = pd.read_csv("data/synthetic_clients.csv")

    features = [
        "edad", "ingreso_mensual", "antiguedad_empleo_años",
        "num_creditos_activos", "dias_mora_historico",
        "tipo_documento", "monto_solicitado", "tiene_buró_negativo"
    ]
    X = df[features]
    y = df["perfil_riesgo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nReporte de clasificación:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Bajo Riesgo", "Riesgo Medio", "Alto Riesgo"]
    ))

    # Guardar modelo
    with open(os.path.join(MODELS_DIR, "risk_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Modelo guardado en {MODELS_DIR}/risk_model.pkl")
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Smart Onboarding Pipeline — Entrenamiento")
    print("=" * 50)
    entrenar_clasificador_documentos()
    entrenar_modelo_riesgo()
    print("\n✅ Ambos modelos entrenados. Siguiente paso: api/main.py")