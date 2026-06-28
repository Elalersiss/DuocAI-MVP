import os
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

from tools import consultar_reglamentos_duoc, consultar_fechas_calendario
from prompts import AGENT_SYSTEM_PROMPT, QUERY_REFORMULATION_PROMPT
from guardrails import run_guardrails

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("GITHUB_TOKEN")

tools = [consultar_reglamentos_duoc, consultar_fechas_calendario]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    base_url="https://models.inference.ai.azure.com"
).bind_tools(tools)

query_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    base_url="https://models.inference.ai.azure.com"
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    is_offensive: Optional[bool]
    is_prompt_injection: Optional[bool]
    is_off_topic: Optional[bool]   # 3er guardrail: temas fuera del dominio académico


# NODO 1: Seguridad — 3 evaluadores en paralelo
def check_guardrails(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content
    flags = run_guardrails(last_message)
    return {
        "is_offensive":       flags["is_offensive"],
        "is_prompt_injection": flags["is_prompt_injection"],
        "is_off_topic":       flags.get("is_off_topic", False),
    }


# NODO 2: Bloqueo
def blocked_response(state: AgentState) -> AgentState:
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


# NODO 3: LLM Principal
def call_model(state: AgentState) -> AgentState:
    response = llm.invoke(
        [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


# NODO 4: Reformulador de query para MongoDB
def generate_query(state: AgentState) -> AgentState:
    conversation = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"] if m.content
    )
    prompt = [
        SystemMessage(content=QUERY_REFORMULATION_PROMPT),
        SystemMessage(content=f"Conversación:\n{conversation}")
    ]
    search_query   = query_llm.invoke(prompt).content.strip()
    last_message   = state["messages"][-1]
    updated_calls  = [{**tc, "args": {"query": search_query}} for tc in last_message.tool_calls]
    updated_message = AIMessage(
        id=last_message.id,
        content=last_message.content,
        tool_calls=updated_calls
    )
    return {"messages": [updated_message]}


# RUTEO CONDICIONAL
def route_after_guardrails(state: AgentState) -> str:
    if (state.get("is_offensive")
            or state.get("is_prompt_injection")
            or state.get("is_off_topic")):
        return "blocked_response"
    return "agent"


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "generate_query"
    return END


# CONSTRUCCIÓN DEL GRAFO
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("check_guardrails", check_guardrails)
graph.add_node("blocked_response", blocked_response)
graph.add_node("agent",            call_model)
graph.add_node("generate_query",   generate_query)
graph.add_node("tools",            tool_node)

graph.set_entry_point("check_guardrails")

graph.add_conditional_edges(
    "check_guardrails",
    route_after_guardrails,
    {"blocked_response": "blocked_response", "agent": "agent"}
)
graph.add_edge("blocked_response", END)
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"generate_query": "generate_query", END: END}
)
graph.add_edge("generate_query", "tools")
graph.add_edge("tools", "agent")

app = graph.compile()