# ==============================================================================
# 1. CONFIGURACIÓN DE LA IMAGEN BASE
# ==============================================================================
FROM python:3.11-slim

# ==============================================================================
# 2. DEFINICIÓN DEL ENTORNO DE TRABAJO
# ==============================================================================
WORKDIR /app

# ==============================================================================
# 3. GESTIÓN E INSTALACIÓN DE DEPENDENCIAS
# ==============================================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# 4. TRANSFERENCIA DE CÓDIGO FUENTE
# ==============================================================================
# Archivos principales del agente
COPY app.py .
COPY agent.py .
COPY prompts.py .
COPY tools.py .
COPY guardrails.py .
COPY observability.py .

# Directorio de datos (contiene el calendario JSON y normativas)
COPY data/ ./data/

# ==============================================================================
# 5. PERSISTENCIA DE OBSERVABILIDAD (EV3)
# ==============================================================================
# Creamos el directorio donde SQLite generará la base de datos en runtime
RUN mkdir -p /app/data/db

# ==============================================================================
# 6. CONFIGURACIÓN DE RED Y EJECUCIÓN
# ==============================================================================
EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]