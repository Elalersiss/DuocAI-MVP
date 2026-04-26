# 1. Imagen base oficial de Python (ligera y optimizada)
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar el archivo de dependencias primero (aprovecha la caché de capas de Podman)
COPY requirements.txt .

# 4. Instalar todas las librerías necesarias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código fuente y los datos (app.py, ingest.py, carpeta data/)
COPY . .

# 6. Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# 7. Comando de ejecución al encender el contenedor (0.0.0.0 permite acceso desde fuera del contenedor)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]