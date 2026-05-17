# ==============================================================================
# 1. CONFIGURACIÓN DE LA IMAGEN BASE
# ==============================================================================
# Selección del entorno de ejecución oficial de Python en su versión estable y reducida
FROM python:3.11-slim

# ==============================================================================
# 2. DEFINICIÓN DEL ENTORNO DE TRABAJO
# ==============================================================================
# Creación y establecimiento del directorio raíz para la aplicación interna del contenedor
WORKDIR /app

# ==============================================================================
# 3. GESTIÓN E INSTALACIÓN DE DEPENDENCIAS
# ==============================================================================
# Transferencia aislada del manifiesto de librerías para optimizar la caché de capas de compilación
COPY requirements.txt .

# Ejecución del gestor de paquetes pip omitiendo el almacenamiento de archivos temporales
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# 4. TRANSFERENCIA DE CÓDIGO FUENTE Y RECURSOS
# ==============================================================================
# Copia estructurada del árbol del proyecto incluyendo scripts lógicos, automatizaciones y datos
COPY . .

# ==============================================================================
# 5. CONFIGURACIÓN DE RED Y REDIRECCIÓN DE PUERTOS
# ==============================================================================
# Declaración de apertura del puerto lógico asignado para la comunicación del servicio
EXPOSE 8501

# ==============================================================================
# 6. DECLARACIÓN DEL COMANDO DE INICIALIZACIÓN
# ==============================================================================
# Punto de entrada para levantar la interfaz Streamlit enlazando el mapeo de red de escucha global
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]