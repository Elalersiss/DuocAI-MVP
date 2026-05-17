import streamlit as st
from main import ejecutar_agente_duoc

# ==============================================================================
# 1. CONFIGURACIÓN DE LA INTERFAZ DE USUARIO (FRONT-END)
# ==============================================================================
# Inicialización de las propiedades del sitio web y descripción base del servicio
st.set_page_config(page_title="DuocAI - Sistema Agéntico", page_icon="🤖")
st.title("DuocAI: Asistente con Agentes Autónomos")
st.markdown("""
    Bienvenido. Este sistema utiliza un **Agente Investigador** y un **Asesor Estudiantil** orquestados jerárquicamente para responder tus dudas académicas y de becas.
""")

# ==============================================================================
# 2. GESTIÓN DE PERSISTENCIA E HISTORIAL DE CONVERSACIÓN
# ==============================================================================
# Inicialización del estado de sesión para el almacenamiento persistente de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Renderizado síncrono de los componentes de chat almacenados históricamente
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==============================================================================
# 3. INTERACCIÓN Y ENLACE CON EL FLUJO MULTI-AGENTE (CORE)
# ==============================================================================
# Captura del input del usuario a través del componente de entrada de texto nativo
if user_input := st.chat_input("Escribe tu consulta aquí..."):
    
    # Procesamiento y visualización inmediata del mensaje emitido por el estudiante
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Contenedor de respuesta del asistente con bloqueador dinámico (spinner)
    with st.chat_message("assistant"):
        with st.spinner("Los agentes están colaborando en tu respuesta..."):
            try:
                # Invocación de la capa lógica y orquestación jerárquica de CrewAI
                respuesta_agentes = ejecutar_agente_duoc(user_input)
                
                # Renderizado de la síntesis final generada por el asesor estudiantil
                st.markdown(respuesta_agentes)
                
                # Persistencia de la respuesta del sistema en el historial de sesión
                st.session_state.messages.append({"role": "assistant", "content": respuesta_agentes})
            
            except Exception as e:
                # Control, captura y despliegue visual de excepciones en la interfaz gráfica
                st.error(f"Error en la orquestación: {e}")