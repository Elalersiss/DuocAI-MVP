import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

# ==============================================================================
# 1. CONFIGURACIÓN DEL ENTORNO Y CONTEXTO DE BASE DE DATOS
# ==============================================================================
# Carga de variables de entorno e inicialización de constantes de infraestructura
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "duocai_db"
COLLECTION_NAME = "duoc_normativas" 
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index" 

# ==============================================================================
# 2. PIPELINE DE EXTRACCIÓN, TRANSFORMACIÓN Y CARGA (ETL MAINSTREAM)
# ==============================================================================
def main():
    """
    Ejecuta el pipeline de ingeniería de datos automatizado (ETL).
    
    Establece conexión con MongoDB Atlas, realiza una purga atómica preventiva,
    extrae texto local aplicando clasificación heurística de metadatos,
    fragmenta semánticamente los documentos y los indexa en el clúster vectorial.
    """
    print("Iniciando pipeline de ingesta de datos (ETL)...")

    # Conexión síncrona al clúster de base de datos en la nube
    try:
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        print("Conexión a MongoDB Atlas establecida.")
    except Exception as e:
        print(f"Error crítico conectando a MongoDB: {e}")
        return

    # Sincronización preventiva para evitar colisión o duplicación de vectores
    print("Ejecutando purga de la base de datos anterior...")
    resultado_borrado = collection.delete_many({})
    print(f"Se eliminaron {resultado_borrado.deleted_count} fragmentos obsoletos.")

    # Lectura del directorio local de almacenamiento documental
    print("Extrayendo texto desde el directorio local 'data/'...")
    docs_totales = [] 
    
    for archivo in os.listdir("data"):
        ruta = os.path.join("data", archivo)
        
        # Clasificación heurística para la inyección programática de metadatos
        categoria_asignada = "financiamiento" if "beca" in archivo.lower() or "financiamiento" in archivo.lower() else "academico"
        docs_cargados = []
        
        # Enrutamiento dinámico de cargadores según la extensión del archivo fuente
        if archivo.endswith(".docx"):
            print(f"Procesando Word: {archivo} | Categoría de Metadato: {categoria_asignada}")
            docs_cargados = Docx2txtLoader(ruta).load()
        elif archivo.endswith(".pdf"):
            print(f"Procesando PDF: {archivo} | Categoría de Metadato: {categoria_asignada}")
            docs_cargados = PyMuPDFLoader(ruta).load()
        elif archivo.endswith(".txt"):
            print(f"Procesando TXT: {archivo} | Categoría de Metadato: {categoria_asignada}")
            docs_cargados = TextLoader(ruta, encoding="utf-8").load()
            
        # Inyección estructural del campo metadato para optimización del RAG
        for doc in docs_cargados:
            doc.metadata["categoria"] = categoria_asignada
            
        docs_totales.extend(docs_cargados) 
    
    # Validación de control de calidad sobre la fase de carga física
    if not docs_totales:
        print("Error de Extracción: No se encontraron documentos válidos en el directorio 'data/'.")
        return
    
    print(f"Se consolidaron {len(docs_totales)} páginas/documentos en total.")

    # Fragmentación recursiva para preservación del contexto legal y de párrafos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,    # Extensión de caracteres por fragmento contextual
        chunk_overlap=300   # Margen de redundancia histórica del veinticinco por ciento
    )
    splits = text_splitter.split_documents(docs_totales)
    print(f"Documentos fragmentados en {len(splits)} chunks contextuales.")

    # Inicialización local y vectorización densa mediante HuggingFace
    print("Inicializando modelo de embeddings multilingüe...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Carga y almacenamiento del set de vectores densos en el indexador de Atlas
    print("Sincronizando vectores y metadatos con MongoDB Atlas...")
    MongoDBAtlasVectorSearch.from_documents(
        documents=splits,
        embedding=embeddings,
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )
    
    print("¡Pipeline ETL completado con éxito! Arquitectura de datos lista para consultas.")

# ==============================================================================
# 3. DISPARADOR DE EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    main()