import os
from dotenv import load_dotenv
import streamlit as st
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL Y VARIABLES DE ENTORNO
# Se cargan las credenciales (API Keys, URIs) desde el archivo .env para 
# garantizar la seguridad de las conexiones externas.
# ==============================================================================
load_dotenv()

# ==============================================================================
# 2. CONFIGURACIÓN DE LA INTERFAZ DE USUARIO (FRONTEND)
# Inicialización de la aplicación web utilizando Streamlit.
# ==============================================================================
st.set_page_config(page_title="DuocAI - Asistente RAG", page_icon="📚")
st.title("DuocAI: Asistente de Reglamentos")
st.markdown("¡Hola! Soy tu asistente de estudio. Pregúntame sobre las normativas institucionales.")

# 2.1. Panel Lateral (Sidebar) para Filtros de Metadatos
st.sidebar.title("⚙️ Filtros de Búsqueda")
st.sidebar.markdown("Optimiza la precisión del modelo aislando el contexto de búsqueda.")
filtro_usuario = st.sidebar.selectbox(
    "Seleccione el dominio de documentos:", 
    ["Búsqueda Global (Todo)", "Solo Financiamiento y Becas", "Solo Reglamentos Académicos"]
)

# ==============================================================================
# 3. GESTIÓN DE ESTADO DE LA SESIÓN (SESSION STATE)
# Mantiene el historial de la conversación en la memoria temporal de la app 
# para renderizar los mensajes en cada recarga de la pantalla.
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Renderizar el historial de chat en la interfaz
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==============================================================================
# 4. INICIALIZACIÓN DEL PIPELINE RAG (CACHÉ DE RECURSOS)
# Se utiliza @st.cache_resource para instanciar las conexiones pesadas a la 
# base de datos y los modelos de IA una sola vez, optimizando el rendimiento.
# ==============================================================================
@st.cache_resource
def init_rag_pipeline():
    # 4.1. Conexión a la base de datos vectorial (MongoDB Atlas)
    client = MongoClient(os.getenv("MONGO_URI"))
    collection = client["duocai_db"]["duoc_normativas"] 
    
    # 4.2. Inicialización del modelo de Embeddings multilingüe
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 4.3. Configuración del motor de búsqueda vectorial
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index"
    )
    
    # 4.4. Instanciación del Modelo de Lenguaje (LLM) vía GitHub Models
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com", 
        temperature=0.0 # Temperatura 0 para evitar alucinaciones en normativas
    )
    
    # 4.5. Configuración del Prompt del Sistema (Ingeniería de Prompts)
    system_prompt = (
        "Eres DuocAI, el asistente virtual oficial de normativas de Duoc UC.\n"
        "A continuación se te proporcionan fragmentos de texto recuperados de la base de datos oficial.\n\n"
        "---------------------\n"
        "CONTEXTO RECUPERADO:\n"
        "{context}\n"
        "---------------------\n\n"
        "INSTRUCCIONES ESTRICTAS:\n"
        "1. Responde basándote ÚNICAMENTE en el CONTEXTO RECUPERADO. Si el contexto menciona varios documentos, "
        "PRIORIZA la información del 'Folleto de Financiamiento' y el 'Reglamento General de Becas' para consultas sobre beneficios económicos.\n"
        "2. Si el contexto NO contiene la información, responde: 'Mis registros normativos actuales no contienen información detallada sobre esa consulta.'\n"
        "3. Haz un listado EXHAUSTIVO de todas las becas y beneficios presentes en el contexto, sin omitir ninguna. Enuméralos claramente indicando sus requisitos y porcentajes de cobertura, e incluye explícitamente los Beneficios Estatales (Gratuidad, CAE, etc.) si aparecen en los fragmentos.\n"
        "4. Si te preguntan qué documentos tienes, responde: Folleto de Financiamiento, Reglamento General de Becas, Reglamento Académico Online y Presencial."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    return vector_store, llm, prompt

# Ejecución de la inicialización con manejo de excepciones
try:
    vector_store, llm, prompt = init_rag_pipeline()
except Exception as e:
    st.error(f"Error crítico al inicializar la arquitectura: {e}")
    st.stop()

# ==============================================================================
# 5. APLICACIÓN DE FILTROS DINÁMICOS Y CONFIGURACIÓN DEL RETRIEVER
# Ajusta los parámetros de búsqueda en MongoDB según la selección del usuario.
# ==============================================================================
search_kwargs = {"k": 15} # Número de fragmentos (chunks) a recuperar

if filtro_usuario == "Solo Financiamiento y Becas":
    search_kwargs["pre_filter"] = {"categoria": {"$eq": "financiamiento"}}
elif filtro_usuario == "Solo Reglamentos Académicos":
    search_kwargs["pre_filter"] = {"categoria": {"$eq": "academico"}}

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs=search_kwargs
)

# ==============================================================================
# 6. FLUJO PRINCIPAL DE INTERACCIÓN Y GENERACIÓN AUMENTADA
# Captura el input del usuario, recupera el contexto, y genera la respuesta.
# ==============================================================================
if user_input := st.chat_input("Escribe tu consulta normativa aquí..."):
    
    # 6.1. Mostrar input del usuario en pantalla
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 6.2. Procesamiento del pipeline RAG
    with st.chat_message("assistant"):
        with st.spinner("Consultando normativas en la base de datos institucional..."):
            
            # A. Recuperación de documentos relevantes (Retrieval)
            docs_recuperados = retriever.invoke(user_input)
            
            # B. Ensamblaje del contexto
            contexto_unido = "\n\n".join([doc.page_content for doc in docs_recuperados])
            
            # C. Inyección de contexto al LLM y generación de respuesta (Generation)
            prompt_listo = prompt.invoke({"context": contexto_unido, "input": user_input})
            respuesta_llm = llm.invoke(prompt_listo)
            answer = respuesta_llm.content
            
            # D. Renderizar respuesta
            st.markdown(answer)

    # -------------------------------------------------------------------------
    # BLOQUE DE DEPURACIÓN EN TERMINAL (Trazabilidad para el desarrollador)
    # -------------------------------------------------------------------------
    print(f"\n--- LOG: RECUPERACIÓN MONGODB (Filtro: {filtro_usuario}) ---")
    for i, doc in enumerate(docs_recuperados):
        print(f"\n[Fragmento {i+1}]: {doc.page_content[:200]}...") 
    print("-------------------------------------------------------------\n")

    # 6.3. Guardar la respuesta del modelo en el historial de sesión
    st.session_state.messages.append({"role": "assistant", "content": answer})