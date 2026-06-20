FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Usuario no-root (DevSecOps)
RUN useradd -m -u 1000 appuser

# Dependencias Python
COPY requirements.txt .
RUN pip install uv && uv pip install --system -r requirements.txt

# Copiar proyecto
COPY . .

# Permisos
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]