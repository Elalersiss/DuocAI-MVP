# DuocAI: Asistente RAG de Normativas Institucionales 📚

**Desarrollado por:** Alexis Margas
**Asignatura:** Ingeniería de Soluciones con IA (ISY0101)

## Descripción del Proyecto
DuocAI es una solución basada en Inteligencia Artificial diseñada para optimizar la consulta y recuperación de normativas institucionales (Reglamentos Académicos y de Financiamiento). Utiliza una arquitectura RAG (Retrieval-Augmented Generation) para garantizar que las respuestas del modelo estén 100% fundamentadas en documentos oficiales, eliminando alucinaciones y proporcionando trazabilidad exacta de la información.

## Arquitectura Técnica
- **Frontend:** Interfaz interactiva desarrollada en `Streamlit`.
- **Pipeline ETL & Chunking:** Extracción de documentos (PDF, DOCX, TXT) y fragmentación semántica usando `LangChain` (RecursiveCharacterTextSplitter).
- **Embeddings:** Vectorización multilingüe local mediante `HuggingFace` (*paraphrase-multilingual-MiniLM-L12-v2*).
- **Base de Datos Vectorial:** Almacenamiento e indexación de alta dimensión en `MongoDB Atlas Vector Search` con filtrado dinámico por metadatos.
- **LLM:** Inferencia mediante `GPT-4o` / `GPT-4o-mini` consumido a través de la API de GitHub Models.
- **Despliegue:** Empaquetado y containerización mediante `Podman` / `Docker`.

## Instrucciones de Despliegue (Para Evaluadores)

Para ejecutar este sistema en un entorno local aislado, asegúrese de tener **Podman** (o Docker) instalado y funcionando.

### 1. Configuración de Variables de Entorno
Clonar el repositorio y crear un archivo llamado `.env` en la raíz del proyecto, tomando como referencia el archivo `.env.example`. Inyecte sus propias credenciales:
```text
MONGO_URI=mongodb+srv://<su-usuario>:<su-password>@<su-cluster>.mongodb.net/
GITHUB_TOKEN=<su-personal-access-token-de-github>
