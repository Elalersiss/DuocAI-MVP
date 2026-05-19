# DuocAI: Asistente RAG Multi-Agente de Normativas Institucionales 📚

**Desarrollado por el equipo:** Alexis Margas 
**Carrera:** Ingeniería en Informática, Duoc UC  
**Asignatura:** Ingeniería de Soluciones con Inteligencia Artificial (ISY0101-002D)  
**Profesor Guía:** Francisco Andrés Macaya Matas  

---

## 📖 Descripción del Proyecto

DuocAI es una solución empresarial basada en Inteligencia Artificial diseñada para optimizar y centralizar la consulta de normativas institucionales, reglamentos académicos y políticas de financiamiento dentro de Duoc UC. 

El sistema evoluciona el paradigma clásico de Recuperación Aumentada por Generación (RAG) hacia una **Arquitectura Híbrida Multi-Agente Jerárquica**. Esto garantiza que las consultas de los estudiantes se respondan de forma fluida y empática, manteniendo un control estricto de la verdad fáctica mediante el aislamiento de dos capas de datos diferenciadas: una capa probabilística para textos normativos ambiguos y una capa determinista local para plazos cronológicos rígidos, eliminando por completo las alucinaciones algorítmicas.

---

## 🏗️ Arquitectura Técnica y Componentes

El ecosistema del software se encuentra modularizado en las siguientes capas tecnológicas integradas:

- **Capa de Presentación (Front-End):** Interfaz gráfica interactiva y conversacional construida en `Streamlit` (`app.py`), diseñada para mitigar la carga cognitiva del usuario y mantener un historial persistente de sesión.
- **Orquestación Cognitiva (CrewAI Hierarchical):** Motor multi-agente en `main.py` gobernado de forma jerárquica por un modelo supervisor (`manager_llm`). Este descompone consultas complejas y coordina las tareas del **Agente Investigador Normativo** (auditor técnico de documentos) y el **Agente Asesor de Apoyo al Estudiante** (traductor de lenguaje normativo a un tono amigable e institucional).
- **Capa de Recuperación Híbrida (Capa de Datos Desacoplada):**
  - *Recuperación Probabilística RAG:* Herramienta dedicada (`consultar_reglamentos_duoc`) que ejecuta búsquedas de registros no estructurados sobre embeddings densos vectorizados.
  - *Recuperación Determinista Local:* Herramienta dedicada (`consultar_fechas_calendario`) que realiza accesos físicos directos al objeto estructurado `Calendario-Académico-Base-2026_v7.json` empleando rutas absolutas dinámicas calculadas en tiempo de ejecución.
- **Capa de Infraestructura y Servicios Cloud:**
  - *Base de Datos Vectorial:* Clúster en la nube `MongoDB Atlas Vector Search` indexado mediante algoritmos de similitud de coseno.
  - *Modelos de Lenguaje (LLMs):* Motor de inferencia `gpt-4o-mini` consumido a través del endpoint de alto rendimiento de `GitHub Models`.
  - *Embeddings:* Modelo multilingüe local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`langchain_huggingface`), con un consumo de API equivalente a costo cero.
- **Capa de Observabilidad (Enterprise Tracing):** Sistema de auditoría integrado mediante el SDK nativo de `LangSmith`. Intercepta los payloads de entrada/salida y las latencias de ejecución agéntica.

---

## 🛠️ Requisitos Previos y Configuración de MongoDB Atlas

Para que el motor híbrido pueda almacenar los embeddings generados por el pipeline ETL y realizar la vectorización síncrona en producción, es obligatorio aprovisionar un clúster en la nube.

### 1. Creación de Cuenta, Proyecto y Clúster
1. **Registro:** Ingrese a la consola oficial de [MongoDB Atlas](https://cloud.mongodb.com/) e inicie sesión.
2. **Creación del Proyecto:** En el menú desplegable superior izquierdo, seleccione **View All Projects**, presione **New Project**, asigne el nombre `DuocAI_Proyecto` y complete la creación.
3. **Despliegue del Clúster:** Ingrese al proyecto y haga clic en **Create Cluster**. Seleccione la capa gratuita dedicada (*M0 Free Tier*) en el proveedor de infraestructura y región geográfica de su preferencia.
4. **Acceso de Seguridad de Red (Network Access):** En el menú lateral izquierdo, bajo la categoría *Security*, acceda a **Network Access**. Haga clic en **Add IP Address** y seleccione **Add Current IP Address** (`0.0.0.0/0`).
5. **Credenciales de Acceso (Database Access):** En la sección **Database Access**, cree un usuario administrador de base de datos (`Atlas Admin`) y configure una contraseña estricta.
6. **Extracción de la URI:** Diríjase a **Database** en la sección de *Deployment*, presione el botón **Connect** del clúster activo, seleccione la opción **Drivers** y copie la cadena de conexión provista (`MONGO_URI`).

### 2. Creación del Índice de Búsqueda Vectorial (Vector Search Index)
Una vez ejecutado el pipeline de ingesta por primera vez, es mandatorio instanciar el índice matemático en Atlas:

1. En la consola de MongoDB Atlas, vaya al menú izquierdo y haga clic en **Search & Vector Search**.
2. Presione el botón **Create Search Index**.
3. Bajo las opciones de *Search Type*, seleccione rigurosamente **Vector Search**.
4. En el campo **Index Name**, asigne de forma estricta el identificador **`vector_index`**.
5. En el menú desplegable de mapeo de colecciones, seleccione la base de datos `duocai_db` y apunte a la colección **`duoc_normativas`**.
6. En el apartado de *Configuration Method*, marque la opción **JSON Editor** y presione *Next*.
7. En el bloque de edición de código, reemplace el contenido existente por el siguiente esquema JSON estructurado:

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

8. Haga clic en **Next** y finalice presionando **Create Vector Search Index**.

### 📐 Notas Técnicas de la Configuración
- **`numDimensions: 384`:** Coincide con la dimensión del espacio vectorial generado por el modelo local. Un desfase en este valor generará una excepción inmediata.
- **`similarity: "cosine"`:** Aplica la métrica de similitud de coseno para evaluar la proximidad angular de los vectores.
- **`path: "categoria"`:** Habilita un nodo de tipo `filter` que faculta a los agentes para discriminar búsquedas estructuradas cruzadas analizando los metadatos inyectados (`financiamiento` o `academico`).

---

## 📦 Instrucciones de Despliegue y Contenerización

El entorno de desarrollo se encuentra empaquetado para aislarse de forma hermética utilizando tecnologías de contenerización sin privilegios de root (`Podman`) o Docker tradicional.

### 1. Configuración de Variables de Entorno
Clone el repositorio en su máquina local. Renombre el archivo de referencia `.env.example` a **`.env`** en la raíz del proyecto e inyecte sus credenciales operativas reales:

```text
MONGO_URI=mongodb+srv://<usuario_atlas>:<password_atlas>@<cluster_uri>.mongodb.net/
GITHUB_TOKEN=ghp_TuTokenPersonalDeGitHubModelsAqui
LANGCHAIN_API_KEY=lsv_TuTokenDeObservabilidadLangSmithAqui
LANGCHAIN_PROJECT=DuocAI_Proyecto
```

---

## 🚀 Guía de Ejecución del Proyecto paso a paso

Una vez configurado el archivo `.env`, puede inicializar el sistema de datos y la interfaz gráfica web seleccionando una de las siguientes dos opciones de despliegue:

### 1. Opción A: Despliegue Automatizado mediante Contenedores (Podman / Docker)
*Esta opción es la recomendada para entornos productivos, ya que compila de forma aislada las dependencias binarias necesarias (`PyMuPDF`, `sentence-transformers`) sin requerir la instalación local de Python.*

- **Construcción de la Imagen del Sistema:** Compile la imagen base utilizando la configuración del manifiesto de capas local:
  ```bash
  podman build -t duocai-app .
  ```
- **Ejecución del Pipeline ETL de Ingesta (Obligatorio la primera vez):** Inicie el contenedor temporal para purgar la base de datos anterior, leer los reglamentos locales de la carpeta `/data`, fragmentar los textos e indexar los vectores en MongoDB Atlas:
  ```bash
  podman run --rm --env-file .env duocai-app python ingest.py
  ```
- **Despliegue del Servidor Web de la Aplicación:** Levante el contenedor en segundo plano exponiendo el puerto nativo de comunicación de red:
  ```bash
  podman run -p 8501:8501 --env-file .env duocai-app
  ```

### 2. Opción B: Despliegue en Entorno de Desarrollo Local Python
*Esta opción requiere tener instalado Python 3.11 en el sistema anfitrión.*

- **Aislamiento del Entorno Virtual:** Instancie un entorno virtual limpio para prevenir conflictos de dependencias con el sistema operativo:
  ```bash
  python -m venv venv
  ```
- **Activación del Entorno:** Active el entorno virtual dependiendo de su sistema operativo:
  - *Sistemas Windows (PowerShell/CMD):*
    ```bash
    venv\Scripts\activate
    ```
  - *Sistemas macOS / Linux (Terminal):*
    ```bash
    source venv/bin/activate
    ```
- **Instalación Determinista de Librerías:** Instale los componentes y dependencias estrictas declaradas en el manifiesto:
  ```bash
  pip install -r requirements.txt
  ```
- **Ejecución de Ingesta de Datos (ETL Local):** Corra el script de orquestación de datos para poblar el clúster vectorial en la nube:
  ```bash
  python ingest.py
  ```
- **Inicialización de la Interfaz Gráfica (Streamlit):** Levante el servidor web local para interactuar con el ecosistema multi-agente:
  ```bash
  streamlit run app.py
  ```

> 🌐 **Nota:** Independiente de la opción de despliegue seleccionada, la aplicación estará disponible en su navegador apuntando a: `http://localhost:8501`

