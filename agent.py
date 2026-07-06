"""
=============================================================================
MÓDULO DE ORQUESTACIÓN AGÉNTICA (LANGGRAPH)
=============================================================================
Este archivo define la arquitectura principal del asistente inteligente.
Utiliza LangGraph para construir un flujo de estado finito (StateGraph) que 
coordina la evaluación de seguridad (guardrails), el razonamiento del LLM, 
la reformulación semántica de consultas y la ejecución de herramientas (RAG y JSON).
"""

import os
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

# Importaciones centrales de LangGraph para la construcción del grafo
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Importaciones de LangChain para el manejo de modelos y mensajes
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

# Importaciones de módulos locales del proyecto
from tools import consultar_reglamentos_duoc, consultar_fechas_calendario
from prompts import AGENT_SYSTEM_PROMPT, QUERY_REFORMULATION_PROMPT
from guardrails import run_guardrails

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL Y LLMs
# ==============================================================================
load_dotenv()

# Configuración del token de acceso para el proveedor del modelo
os.environ["OPENAI_API_KEY"] = os.getenv("GITHUB_TOKEN")

# Lista de herramientas disponibles que el agente puede decidir utilizar
tools = [consultar_reglamentos_duoc, consultar_fechas_calendario]

# LLM Principal: Es el cerebro del agente. Se le "atan" (bind) las herramientas
# para que sepa que existen y decida cuándo invocarlas.
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # Temperatura 0 para respuestas deterministas y precisas
    base_url="https://models.inference.ai.azure.com"
).bind_tools(tools)

# LLM Secundario (Reformulador): Se usa exclusivamente para transformar
# las preguntas conversacionales en consultas optimizadas para la base de datos.
query_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    base_url="https://models.inference.ai.azure.com"
)

# ==============================================================================
# 2. DEFINICIÓN DEL ESTADO DEL GRAFO (MEMORIA)
# ==============================================================================
class AgentState(TypedDict):
    """
    Estructura de datos que fluye a través de todos los nodos del grafo.
    Almacena el historial de la conversación y las banderas de seguridad.
    """
    # 'add_messages' asegura que los mensajes nuevos se concatenen al historial
    messages: Annotated[list, add_messages]
    is_offensive: Optional[bool]
    is_prompt_injection: Optional[bool]
    is_off_topic: Optional[bool]


# ==============================================================================
# 3. DEFINICIÓN DE NODOS (ACCIONES DEL FLUJO)
# ==============================================================================

def check_guardrails(state: AgentState) -> AgentState:
    """
    Nodo 1: Capa de Seguridad de Entrada.
    Intercepta el último mensaje del usuario y lo pasa por evaluadores paralelos
    para detectar contenido malicioso, ofensivo o fuera de contexto.
    """
    last_message = state["messages"][-1].content
    flags = run_guardrails(last_message)
    
    # Actualiza el estado global con los resultados de la evaluación
    return {
        "is_offensive":       flags["is_offensive"],
        "is_prompt_injection": flags["is_prompt_injection"],
        "is_off_topic":       flags.get("is_off_topic", False),
    }

def blocked_response(state: AgentState) -> AgentState:
    """
    Nodo 2: Respuesta de Bloqueo (Contingencia).
    Se ejecuta únicamente si el nodo 'check_guardrails' detecta una anomalía.
    Retorna un mensaje estándar sin consumir tokens del LLM principal.
    """
    return {
        "messages": [
            AIMessage(
                content=(
                    "🛡️ Tu mensaje ha sido bloqueado por las políticas de seguridad de DuocAI. "
                    "Por favor, reformula tu consulta dentro del contexto académico de Duoc UC."
                )
            )
        ]
    }

def call_model(state: AgentState) -> AgentState:
    """
    Nodo 3: Razonamiento Principal del Agente.
    Toma el historial de mensajes, inyecta el prompt del sistema y le pide al LLM
    que decida el siguiente paso (responder o usar una herramienta).
    """
    # Se inyecta la personalidad y reglas del sistema antes del historial de usuario
    response = llm.invoke(
        [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}

def generate_query(state: AgentState) -> AgentState:
    """
    Nodo 4: Interceptor y Reformulador de Búsqueda.
    Si el LLM decide usar una herramienta, este nodo intercepta esa decisión,
    toma el historial conversacional y crea una consulta semántica limpia para
    maximizar la precisión de la base de datos vectorial (RAG).
    """
    # Reconstruye el contexto de la conversación en texto plano
    conversation = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"] if m.content
    )
    
    prompt = [
        SystemMessage(content=QUERY_REFORMULATION_PROMPT),
        SystemMessage(content=f"Conversación:\n{conversation}")
    ]
    
    # 1. Captura la respuesta completa, no solo el texto
    llm_response = query_llm.invoke(prompt)
    search_query = llm_response.content.strip()
    
    # 2. Extrae los metadatos de los tokens de esa respuesta
    tokens_usados = llm_response.usage_metadata
    
    last_message = state["messages"][-1]
    updated_calls = [{**tc, "args": {"query": search_query}} for tc in last_message.tool_calls]
    
    # 3. Inyecta explícitamente los metadatos al reconstruir el mensaje
    updated_message = AIMessage(
        id=last_message.id,
        content=last_message.content,
        tool_calls=updated_calls,
        usage_metadata=tokens_usados 
    )
    return {"messages": [updated_message]}


# ==============================================================================
# 4. RUTEO CONDICIONAL (LÓGICA DE DIRECCIÓN)
# ==============================================================================

def route_after_guardrails(state: AgentState) -> str:
    """
    Decide el camino a tomar inmediatamente después de evaluar la seguridad.
    Si alguna bandera es True, desvía el flujo al nodo de bloqueo.
    De lo contrario, permite el paso al agente principal.
    """
    if (state.get("is_offensive")
            or state.get("is_prompt_injection")
            or state.get("is_off_topic")):
        return "blocked_response"
    
    return "agent"

def should_continue(state: AgentState) -> str:
    """
    Decide si el agente ya tiene la respuesta final o si necesita consultar datos.
    Si el último mensaje contiene una llamada a herramienta, desvía al reformulador.
    Si no, termina la ejecución (END).
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "generate_query"
    
    return END


# ==============================================================================
# 5. CONSTRUCCIÓN Y COMPILACIÓN DEL GRAFO
# ==============================================================================

# Nodo preconstruido de LangGraph que ejecuta físicamente las funciones de tools.py
tool_node = ToolNode(tools)

# Inicialización del grafo con el esquema de memoria definido
graph = StateGraph(AgentState)

# Adición de todos los nodos al grafo
graph.add_node("check_guardrails", check_guardrails)
graph.add_node("blocked_response", blocked_response)
graph.add_node("agent",            call_model)
graph.add_node("generate_query",   generate_query)
graph.add_node("tools",            tool_node)

# Se define el punto de inicio estricto del sistema
graph.set_entry_point("check_guardrails")

# Trazado de las aristas (caminos) condicionales y directos
# Flujo 1: Entrada -> Seguridad -> (Bloqueo o Agente)
graph.add_conditional_edges(
    "check_guardrails",
    route_after_guardrails,
    {"blocked_response": "blocked_response", "agent": "agent"}
)

# Flujo 2: Bloqueo -> Fin
graph.add_edge("blocked_response", END)

# Flujo 3: Agente -> (Fin o Reformular Query)
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"generate_query": "generate_query", END: END}
)

# Flujo 4: Reformular Query -> Ejecutar Tool -> Volver al Agente
graph.add_edge("generate_query", "tools")
graph.add_edge("tools", "agent")

# 1. Instanciamos el checkpointer para la memoria
memory = MemorySaver()

# 2. Compilación final con el checkpointer incluido
app = graph.compile(checkpointer=memory)