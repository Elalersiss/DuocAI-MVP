import os
import json
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from pymongo import MongoClient
from langsmith import traceable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

# Carga de variables de entorno locales
load_dotenv()

# ==============================================================================
# 1. RECURSOS COMPARTIDOS (INICIALIZACIÓN ÚNICA PARA OPTIMIZACIÓN DE LATENCIA)
# ==============================================================================
# Instanciamiento del modelo de embeddings densos multilingües
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Conexión al clúster de base de datos e indexador vectorial en la nube
client = MongoClient(os.getenv("MONGO_URI"))
vector_store = MongoDBAtlasVectorSearch(
    collection=client["duocai_db"]["duoc_normativas"], 
    embedding=embeddings_model, 
    index_name="vector_index"
)

# ==============================================================================
# 2. CONTRATOS DE VALIDACIÓN DE ENTRADA (CONTRATOS DE ESQUEMA EN PYDANTIC V2)
# ==============================================================================
class EntradaReglamentos(BaseModel):
    """Esquema de validación estricta para la herramienta de búsqueda RAG."""
    query: str = Field(..., description="Texto de la consulta para buscar en los reglamentos de Duoc UC.")

class EntradaCalendario(BaseModel):
    """Esquema de validación estricta para la herramienta determinista del calendario."""
    query: str = Field(default="", description="Texto de la consulta o filtro para el calendario académico.")

# ==============================================================================
# 3. CAPA LÓGICA DE PROCESAMIENTO INTERNO (PROCESOS TRAZABLES EN LANGSMITH)
# ==============================================================================
@traceable(run_type="tool", name="ejecutar_rag_reglamentos")
def _proceso_interno_rag(query: str) -> str:
    """
    Ejecuta una búsqueda de recuperación semántica sobre la base de datos documental.
    
    Extrae los k=7 fragmentos vectoriales más cercanos basándose en la métrica 
    de similitud de coseno y consolida los bloques de texto.
    """
    docs = vector_store.as_retriever(search_kwargs={"k": 7}).invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

@traceable(run_type="tool", name="ejecutar_lectura_calendario")
def _proceso_interno_calendario(query: str) -> str:
    """
    Realiza una extracción física determinista sobre el almacenamiento plano estructurado.
    
    Resuelve la localización del archivo JSON mediante cálculo dinámico de la ruta 
    absoluta del sistema de archivos, mitigando excepciones de entorno.
    """
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo = os.path.join(directorio_actual, "data", "Calendario-Académico-Base-2026_v7.json")
    
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2, ensure_ascii=False)

# ==============================================================================
# 4. INTERFACES DE INTEGRACIÓN DE HERRAMIENTAS (CLASES BASE DE CREWAI)
# ==============================================================================
class HerramientaReglamentos(BaseTool):
    """Clase encargada de exponer la capa probabilística RAG al ecosistema agéntico."""
    name: str = "consultar_reglamentos_duoc"
    description: str = (
        "Consulta la base de datos oficial de Duoc UC para resolver dudas sobre "
        "reglamentos académicos, becas y beneficios institucionales."
    )
    args_schema: type[BaseModel] = EntradaReglamentos

    def _run(self, query: str) -> str:
        try:
            return _proceso_interno_rag(query)
        except Exception as e:
            return f"Error en la base de datos: {e}"

class HerramientaCalendario(BaseTool):
    """Clase encargada de exponer la capa determinista de fechas al ecosistema agéntico."""
    name: str = "consultar_fechas_calendario"
    description: str = (
        "Consulta las fechas límites del calendario académico 2026 "
        "(suspensiones, retiros, exámenes, inicio de clases)."
    )
    args_schema: type[BaseModel] = EntradaCalendario

    def _run(self, query: str) -> str:
        try:
            return _proceso_interno_calendario(query)
        except Exception as e:
            return f"Error al leer el calendario académico: {e}"

# ==============================================================================
# 5. INSTANCIAMIENTO DE COMPONENTES EXPORTABLES
# ==============================================================================
# Objetos finales expuestos e inyectados en la definición de agentes autónomos
herramienta_duoc = HerramientaReglamentos()
herramienta_calendario = HerramientaCalendario()