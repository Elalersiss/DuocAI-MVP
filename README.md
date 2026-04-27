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
```
## 2. Requisitos Previos y Configuración de MongoDB Atlas

Para que el motor RAG de DuocAI pueda almacenar los embeddings y realizar consultas semánticas, es necesario configurar un entorno en la nube utilizando MongoDB Atlas. Siga estos pasos detallados para inicializar su base de datos.

### 3. Creación de Cuenta, Proyecto y Clúster

1.  **Registro:** Ingrese a [MongoDB Atlas](https://cloud.mongodb.com/) y cree una cuenta gratuita (o inicie sesión si ya posee una).
2.  **Creación del Proyecto:** En la esquina superior izquierda de la interfaz, abra el menú desplegable y seleccione **View All Projects**. Haga clic en **New Project**, asígnele un nombre (por ejemplo, *DuocAI_Proyecto*) y créelo.
3.  **Creación del Clúster:** Ingrese al proyecto recién creado. Haga clic en **Create Cluster** para desplegar un nuevo clúster. Puede seleccionar la capa gratuita (M0 Free Tier) en el proveedor de nube de su preferencia.
4.  **Acceso de Red (Network Access):** En el menú lateral izquierdo, vaya a la sección **Network Access** bajo *Security*. Haga clic en *Add IP Address* y seleccione *Allow Access From Anywhere* (`0.0.0.0/0`) para permitir que el contenedor local se conecte sin bloqueos de firewall.
5.  **Acceso a la Base de Datos (Database Access):** En la sección **Database Access**, cree un nuevo usuario de base de datos con una contraseña segura. Guarde estas credenciales.
6.  **Obtener la URI de Conexión:** Vaya a **Database** bajo *Deployment*, haga clic en el botón **Connect** de su clúster, seleccione **Drivers** y copie la cadena de conexión (`MONGO_URI`). Esta es la que deberá inyectar en su archivo `.env`.

### 4. Configuración de Búsqueda Vectorial (Vector Search Index)

Una vez que el clúster esté creado y haya ejecutado el proyecto por primera vez (para que la base de datos y la colección se creen automáticamente mediante la ingesta), es obligatorio configurar un índice de búsqueda vectorial:

1.  Acceda a su cuenta en [MongoDB Atlas](https://cloud.mongodb.com/).
2.  En el menú lateral izquierdo, diríjase a **Search & Vector Search** en la categoría DATABASE.
3.  Haga clic en el botón **Create Search Index**.
4.  Dentro de las opciones de **Search Type**, seleccione **Vector Search**.
5.  En el apartado de **Index Name**, debe poner el nombre de **`vector_index`**.
6.  Un poco mas abajo, en el menu desplegable que llevara el nombre de la base de datos, seleccione la colección donde hizo la ingesta, que por default viene como **`duoc_normativas`**.
7.  Dentro de las opciones de **Configuration Method**, seleccione **JSON Editor** y presione *Next*.
9.  En el editor de texto, pegue la siguiente configuración JSON:

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
      "path": "metadata.domain",
      "type": "filter"
    }
  ]
}
```

### Notas Técnicas de la Configuración

* **`numDimensions: 384`**: Corresponde a la dimensión del modelo `paraphrase-multilingual-MiniLM-L12-v2` utilizado en el pipeline RAG. Es fundamental que este valor coincida con la salida del modelo de embeddings para evitar errores de indexación.
* **`similarity: "cosine"`**: Utiliza la similitud de coseno para medir la relevancia semántica de los documentos recuperados. Este algoritmo es ideal para comparar vectores de texto, ya que se enfoca en la orientación (contexto) más que en la magnitud.
* **`metadata.domain`**: Configurado como tipo `filter` para permitir la segmentación estricta de los documentos institucionales. Esto habilita al sistema para discriminar búsquedas entre distintos reglamentos (Académico, Becas, etc.) de forma eficiente.

10.  Finalmente da en **Next** y en **Create Vector Search Index** para crear el index.
