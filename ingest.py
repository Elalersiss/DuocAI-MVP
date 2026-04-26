import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL Y VARIABLES DE ENTORNO
# Carga de credenciales y configuración estática de la arquitectura de datos.
# ==============================================================================
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "duocai_db"
COLLECTION_NAME = "duoc_normativas" 
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index" 

def main():
    print("Iniciando pipeline de ingesta de datos (ETL)...")

    # ==============================================================================
    # 2. CONEXIÓN A LA BASE DE DATOS (MONGODB ATLAS)
    # Establece la conexión con el clúster en la nube utilizando el URI seguro.
    # ==============================================================================
    try:
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        print("Conexión a MongoDB Atlas establecida.")
    except Exception as e:
        print(f"Error crítico conectando a MongoDB: {e}")
        return

    # ==============================================================================
    # 3. PURGA DE DATOS EXISTENTES (ETL CLEANUP)
    # Elimina los documentos previos en la colección para evitar duplicidad de 
    # vectores, manteniendo intacta la configuración del índice de búsqueda.
    # ==============================================================================
    print("Ejecutando purga de la base de datos anterior...")
    resultado_borrado = collection.delete_many({})
    print(f"Se eliminaron {resultado_borrado.deleted_count} fragmentos obsoletos.")

    # ==============================================================================
    # 4. EXTRACCIÓN Y ASIGNACIÓN DE METADATOS (DATA EXTRACTION)
    # Recorre el directorio de origen, identifica el formato del documento y 
    # aplica el loader correspondiente, inyectando metadatos para el filtrado RAG.
    # ==============================================================================
    print("Extrayendo texto desde el directorio local 'data/'...")
    
    docs_totales = [] 
    
    for archivo in os.listdir("data"):
        ruta = os.path.join("data", archivo)
        
        # Clasificación heurística basada en la nomenclatura del archivo
        categoria_asignada = "financiamiento" if "beca" in archivo.lower() or "financiamiento" in archivo.lower() else "academico"
        
        docs_cargados = []
        
        # Selección dinámica de la herramienta de extracción según extensión
        if archivo.endswith(".docx"):
            print(f"Procesando Word: {archivo} | Categoría de Metadato: {categoria_asignada}")
            docs_cargados = Docx2txtLoader(ruta).load()
        elif archivo.endswith(".pdf"):
            print(f"Procesando PDF: {archivo} | Categoría de Metadato: {categoria_asignada}")
            docs_cargados = PyMuPDFLoader(ruta).load()
        elif archivo.endswith(".txt"):
            print(f"Procesando TXT: {archivo} | Categoría de Metadato: {categoria_asignada}")
            docs_cargados = TextLoader(ruta, encoding="utf-8").load()
            
        # Inyección del campo 'categoria' en los metadatos del objeto Document
        for doc in docs_cargados:
            doc.metadata["categoria"] = categoria_asignada
            
        # Consolidación estructural en la lista maestra
        docs_totales.extend(docs_cargados) 
    
    # Validación de integridad de la fase de extracción
    if not docs_totales:
        print("Error de Extracción: No se encontraron documentos válidos en el directorio 'data/'.")
        return
    
    print(f"Se consolidaron {len(docs_totales)} páginas/documentos en total.")

    # ==============================================================================
    # 5. FRAGMENTACIÓN DE DOCUMENTOS (CHUNKING)
    # Divide los textos largos en fragmentos procesables (chunks) para preservar
    # el contexto semántico y respetar la ventana de tokens del LLM.
    # ==============================================================================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,    # Tamaño óptimo para capturar secciones normativas completas
        chunk_overlap=300   # Solapamiento para garantizar la cohesión entre párrafos
    )
    splits = text_splitter.split_documents(docs_totales)
    print(f"Documentos fragmentados en {len(splits)} chunks contextuales.")

    # ==============================================================================
    # 6. GENERACIÓN DE EMBEDDINGS (VECTORIZACIÓN)
    # Transforma el texto humano en representaciones vectoriales multidimensionales.
    # ==============================================================================
    print("Inicializando modelo de embeddings multilingüe...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ==============================================================================
    # 7. INGESTA EN LA BASE DE DATOS VECTORIAL (LOAD)
    # Sube los fragmentos transformados y sus vectores a la colección en la nube.
    # ==============================================================================
    print("Sincronizando vectores y metadatos con MongoDB Atlas...")
    MongoDBAtlasVectorSearch.from_documents(
        documents=splits,
        embedding=embeddings,
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )
    
    print("¡Pipeline ETL completado con éxito! Arquitectura de datos lista para consultas.")

if __name__ == "__main__":
    main()