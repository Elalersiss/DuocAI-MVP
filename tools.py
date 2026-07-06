import os
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from langsmith import traceable
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL Y CONEXIONES
# ==============================================================================
load_dotenv()

# Instanciamiento del modelo de embeddings densos multilingües (HuggingFace)
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Conexión al clúster de base de datos e indexador vectorial en la nube
client = MongoClient(os.getenv("MONGO_URI"))
collection = client["duocai_db"]["duoc_normativas"]

# ==============================================================================
# 2. LÓGICA DE BÚSQUEDA RAG (CON TRAZABILIDAD EN LANGSMITH)
# ==============================================================================
@traceable(name="rag_retrieve")
def retrieve(query: str, top_k: int = 8) -> list[dict]:
    """
    Realiza la búsqueda vectorial en MongoDB Atlas y devuelve los fragmentos más relevantes.
    La etiqueta @traceable permite que LangSmith mida exactamente cuánto tarda y qué busca esta función.
    """
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection, 
        embedding=embeddings_model, 
        index_name="vector_index"
    )
    
    # Realizamos la búsqueda de similitud semántica
    docs = vector_store.similarity_search(query, k=top_k)
    
    # Retornamos una lista de diccionarios con el texto y metadatos
    return [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]

# ==============================================================================
# 3. HERRAMIENTAS EXPUESTAS AL AGENTE (USANDO @tool DE LANGCHAIN)
# ==============================================================================
@tool
def consultar_reglamentos_duoc(query: str) -> str:
    """
    Consulta la base de datos oficial de Duoc UC para resolver dudas sobre 
    reglamentos académicos, becas y beneficios institucionales.
    """
    try:
        docs = retrieve(query)
        if not docs:
            return "No se encontró información relevante en los reglamentos."
        
        # Formateamos los fragmentos para que el LLM los entienda fácilmente y pueda citarlos
        return "\n\n---\n\n".join([f"Fragmento recuperado:\n{doc['text']}" for doc in docs])
    except Exception as e:
        return f"Error en la base de datos al buscar reglamentos: {e}"

@tool
def consultar_fechas_calendario(query: str) -> str:
    """
    Consulta las fechas límites y plazos exactos del calendario académico de este año 
    (ej. suspensión de semestre, retiros, exámenes, inicio de clases).
    """
    try:
        # AQUÍ: Pon la ruta exacta donde tienes guardado tu archivo JSON
        ruta_calendario = "data/Calendario-Académico-Base-2026_v7.json"
        
        if not os.path.exists(ruta_calendario):
            return "Error interno: El archivo del calendario académico no se encuentra en la ruta especificada."
            
        with open(ruta_calendario, 'r', encoding='utf-8') as archivo:
            datos_calendario = json.load(archivo)
            
        # Convertimos el JSON a un texto formateado para que el LLM lo lea fácilmente
        texto_calendario = json.dumps(datos_calendario, indent=2, ensure_ascii=False)
        
        return f"Aquí tienes la información completa del calendario académico. Busca la respuesta a la consulta del usuario aquí:\n{texto_calendario}"
        
    except Exception as e:
        return f"Error al consultar el calendario: {e}"