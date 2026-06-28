# DuocAI: Asistente Académico con Observabilidad 📚

**Desarrollado por:** Alexis Margas  
**Carrera:** Ingeniería en Informática, Duoc UC  
**Asignatura:** Ingeniería de Soluciones con Inteligencia Artificial (ISY0101-002D)  
**Profesor Guía:** Francisco Andrés Macaya Matas  

---

## 📖 Descripción del Proyecto

DuocAI es un asistente académico conversacional basado en Inteligencia Artificial, diseñado para centralizar y democratizar el acceso a la normativa institucional de Duoc UC. El sistema permite a los estudiantes resolver dudas sobre reglamentos académicos, becas, financiamiento y fechas del calendario académico mediante lenguaje natural.

La solución implementa una arquitectura de agente inteligente construida sobre **LangGraph**, combinando un pipeline de Recuperación Aumentada por Generación (RAG) sobre MongoDB Atlas con una capa determinista de consulta de datos estructurados. Incorpora además un sistema completo de **observabilidad con persistencia en SQLite**, guardrails de seguridad en paralelo y un dashboard interactivo de métricas desarrollado en Streamlit.

---

## 🏗️ Arquitectura Técnica

El sistema está modularizado en los siguientes componentes:

- **`app.py`** — Interfaz gráfica conversacional en Streamlit con dashboard de observabilidad integrado. Gestiona el ciclo completo de cada interacción: invocación del grafo, registro de trazas por nodo, evaluación automática de calidad con juez LLM y visualización de métricas históricas.
- **`agent.py`** — Grafo LangGraph con cinco nodos: `check_guardrails → agent → generate_query → tools → agent`. Define el estado, la lógica de routing condicional y la compilación del grafo.
- **`tools.py`** — Herramientas expuestas al agente: `consultar_reglamentos_duoc` (búsqueda vectorial en MongoDB Atlas) y `consultar_fechas_calendario` (lectura determinista del archivo JSON del calendario académico).
- **`guardrails.py`** — Tres evaluadores de seguridad ejecutados en paralelo mediante `ThreadPoolExecutor`: detección de contenido ofensivo, prompt injection y consultas fuera del dominio académico.
- **`prompts.py`** — Prompts del sistema para el agente principal, el reformulador de queries y los tres guardrails.
- **`observability.py`** — Inicialización y operaciones CRUD sobre la base de datos SQLite (`duocai_observability.db`), con cuatro tablas: `sessions`, `messages`, `traces` y `evaluations`.
- **`ingest.py`** — Pipeline ETL para procesar, fragmentar, vectorizar e indexar los documentos normativos en MongoDB Atlas.

---

## 🛠️ Requisitos Previos

### Configuración de MongoDB Atlas

1. Ingrese a [MongoDB Atlas](https://cloud.mongodb.com/) y cree un proyecto llamado `DuocAI_Proyecto`.
2. Despliegue un clúster gratuito **M0 Free Tier**.
3. En **Network Access**, agregue la IP `0.0.0.0/0`.
4. En **Database Access**, cree un usuario con rol `Atlas Admin`.
5. Copie la cadena de conexión desde **Database → Connect → Drivers** (`MONGO_URI`).

### Índice de Búsqueda Vectorial

Una vez ejecutado el pipeline de ingesta por primera vez, cree el índice vectorial:

1. En Atlas, vaya a **Search & Vector Search → Create Search Index**.
2. Seleccione **Vector Search** como tipo.
3. Asigne el nombre exacto **`vector_index`**.
4. Seleccione la base de datos `duocai_db` y la colección `duoc_normativas`.
5. En **JSON Editor**, pegue el siguiente esquema:

```json
{
  "fields": [
    {
      "numDimensions": 384,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "categoria",
      "type": "filter"
    }
  ]
}
```

> **Nota:** `numDimensions: 384` debe coincidir exactamente con la dimensión del modelo `paraphrase-multilingual-MiniLM-L12-v2`. Cualquier desfase generará una excepción en runtime.

---

## ⚙️ Configuración de Variables de Entorno

Cree un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```text
MONGO_URI=mongodb+srv://<usuario>:<password>@<cluster>.mongodb.net/
GITHUB_TOKEN=ghp_TuTokenDeGitHubModelsAqui
```

> El sistema utiliza **GitHub Models** como endpoint de inferencia para `gpt-4o-mini`. El `GITHUB_TOKEN` es el token de acceso personal de GitHub con permisos de modelos.

---

## 🚀 Instrucciones de Ejecución

### Opción A: Contenedor Podman o Docker (Recomendada)

Esta opción no requiere instalar Python ni dependencias en el sistema anfitrión.

**1. Construir la imagen:**
```bash
podman build -t duocai-mvp .
```

**2. Ejecutar el pipeline de ingesta (obligatorio la primera vez):**
```bash
podman run --rm --env-file .env duocai-mvp python ingest.py
```

**3. Levantar la aplicación:**
```bash
podman run -p 8501:8501 --env-file .env duocai-mvp
```

La aplicación estará disponible en: `http://localhost:8501`

---

### Opción B: Entorno Local Python

Requiere Python 3.11 instalado en el sistema.

**1. Crear y activar el entorno virtual:**
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS / Linux:
```bash
source venv/bin/activate
```

**2. Instalar dependencias:**
```bash
pip install -r requirements.txt
```

**3. Ejecutar el pipeline de ingesta (obligatorio la primera vez):**
```bash
python ingest.py
```

**4. Iniciar la aplicación:**
```bash
streamlit run app.py
```

La aplicación estará disponible en: `http://localhost:8501`

---

## 📊 Sistema de Observabilidad (EV3)

El sistema registra automáticamente cada interacción en una base de datos SQLite local. El dashboard de observabilidad, accesible desde la pestaña **Dashboard de Observabilidad (EV3)** dentro de la aplicación, expone las siguientes métricas en tiempo real:

- **Latencia promedio** por consulta y desglosada por nodo del grafo.
- **Precisión** calculada por el juez LLM (Agent Goal Accuracy).
- **Tasa de error** basada en veredictos del evaluador automático.
- **Costo acumulado** estimado en USD según precios de gpt-4o-mini.
- **Activaciones de guardrails** por tipo (ofensivo, prompt injection, off-topic).
- **Frecuencia de uso de herramientas** (RAG vs. calendario).
- **Evolución histórica de latencia** a lo largo del tiempo.
- **Detección automática de cuellos de botella** y recomendaciones de optimización.

> La base de datos `duocai_observability.db` se crea automáticamente al iniciar la aplicación por primera vez. No es necesario ningún paso de configuración adicional.

---
