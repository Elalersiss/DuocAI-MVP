"""
=============================================================================
MÓDULO DE INTERFAZ DE USUARIO Y OBSERVABILIDAD (STREAMLIT)
=============================================================================
Este archivo es el punto de entrada (entrypoint) de la aplicación DuocAI.
Despliega una interfaz web interactiva con dos pestañas principales:
1. Un Chatbot conectado al grafo de LangGraph para resolver consultas.
2. Un Dashboard Analítico que lee de una base de datos SQLite para mostrar 
   métricas de rendimiento, costos y evaluaciones de calidad en tiempo real.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import uuid
import time
import sqlite3
import pandas as pd
import os
import json

# Importación de la lógica del agente y el módulo de persistencia local
from agent import app as agent_graph
from observability import log_session, log_message, log_trace, log_evaluation

# ==============================================================================
# 1. CONFIGURACIÓN GLOBAL DE LA APLICACIÓN
# ==============================================================================
st.set_page_config(
    page_title="DuocAI - Asistente y Observabilidad",
    page_icon="🤖",
    layout="wide"
)
st.title("DuocAI: Asistente Académico con Observabilidad")

# Creación de pestañas modulares para separar la interacción del análisis
tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard Analítico"])

# ==============================================================================
# 2. INICIALIZACIÓN DEL ESTADO DE SESIÓN (MEMORIA VOLÁTIL)
# ==============================================================================
# Se asegura que la memoria de la conversación persista durante la recarga de la página
if "messages" not in st.session_state:
    st.session_state.messages = []

# Se genera un UUID único para identificar la sesión actual en la base de datos
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ==============================================================================
# 3. MOTOR DE EVALUACIÓN DE CALIDAD AUTOMÁTICO (JUEZ LLM)
# ==============================================================================
def evaluate_response_quality(question: str, answer: str) -> dict:
    """
    Actúa como un evaluador autónomo ("LLM-as-a-Judge") que audita 
    la calidad de la respuesta final entregada por el agente principal.
    
    Args:
        question (str): La consulta original del usuario.
        answer (str): La respuesta final generada por el sistema.
        
    Returns:
        dict: Un objeto con el veredicto estructurado:
              - 'verdict' (str): "good", "bad", o "blocked".
              - 'score' (int): Calificación del 1 al 10.
              - 'reason' (str): Justificación en lenguaje natural.
    """
    # Exclusión rápida: Si el mensaje fue bloqueado previamente por un Guardrail, 
    # no se gasta tokens en evaluarlo.
    if not answer or "bloqueado por las políticas" in answer:
        return {"verdict": "blocked", "score": 1, "reason": "Mensaje bloqueado por guardrails de seguridad."}

    try:
        # Se instancia un LLM ligero para la tarea de evaluación
        judge_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,  # Alta determinación
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url="https://models.inference.ai.azure.com",
        )
        
        # Prompt de sistema que exige una respuesta en formato JSON estricto
        prompt = f"""Eres un evaluador de calidad de respuestas de un asistente académico de DuocUC.
Evalúa si la siguiente respuesta del asistente fue útil, relevante y correcta para la pregunta del estudiante.

Pregunta del estudiante: {question}
Respuesta del asistente: {answer}

Responde ÚNICAMENTE con un JSON con exactamente estos campos:
{{
  "verdict": "good" o "bad",
  "score": número entero del 1 al 10,
  "reason": "explicación breve en una oración"
}}
"""
        result = judge_llm.invoke(prompt)
        
        # Limpieza de sintaxis Markdown (```json) para evitar errores de parseo
        raw = result.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
        
    except Exception:
        # Mecanismo de contingencia: Si la API falla, se usa una evaluación basada en palabras clave
        bad_signals = ["no se encontró", "lamentablemente", "no tengo información",
                       "no puedo responder", "error en la base de datos"]
        is_bad = any(s in answer.lower() for s in bad_signals)
        return {
            "verdict": "bad" if is_bad else "good",
            "score": 3 if is_bad else 8,
            "reason": "Evaluación heurística de contingencia (API del Juez inaccesible)."
        }

# ==============================================================================
# 4. LÓGICA DE LA INTERFAZ DE USUARIO (PESTAÑA 1: CHAT)
# ==============================================================================
with tab_chat:
    st.markdown("Bienvenido. Este sistema utiliza arquitectura RAG sobre normativas institucionales de Duoc UC.")

    # 4.1. Renderizado del historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4.2. Captura y procesamiento del Input del Usuario
    if user_input := st.chat_input("Escribe tu consulta aquí..."):
        session_id = st.session_state.thread_id
        
        # Registro inicial de la sesión en la base de datos
        log_session(session_id)
        user_msg_id = str(uuid.uuid4())

        # Muestra el mensaje del usuario en la pantalla
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 4.3. Ejecución del Grafo de Agentes y Trazabilidad (Streaming)
        with st.chat_message("assistant"):
            with st.spinner("Procesando consulta..."):
                try:
                    config = {"configurable": {"thread_id": session_id}}

                    final_answer = ""
                    is_offensive  = 0
                    is_injection  = 0
                    step_order    = 0
                    node_start = time.time()

                    # El modo "updates" permite interceptar el grafo nodo por nodo
                    for chunk in agent_graph.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                        stream_mode="updates"
                    ):
                        node_end  = time.time()
                        duration_ms = (node_end - node_start) * 1000

                        for node_name, state in chunk.items():
                            tool_name = None
                            node_p_tokens = 0
                            node_c_tokens = 0
                            messages = state.get("messages", [])

                            # Captura de banderas de seguridad
                            if node_name == "check_guardrails":
                                is_offensive = 1 if state.get("is_offensive") else 0
                                is_injection = 1 if state.get("is_prompt_injection") else 0

                            # Extracción de metadatos (tokens y herramientas) del último mensaje
                            for msg in messages:
                                if isinstance(msg, AIMessage) and msg.tool_calls:
                                    tool_name = msg.tool_calls[0]["name"]
                                if isinstance(msg, AIMessage) and msg.usage_metadata:
                                    node_p_tokens = msg.usage_metadata.get("input_tokens", 0)
                                    node_c_tokens = msg.usage_metadata.get("output_tokens", 0)
                                if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                                    final_answer = msg.content

                            # Registro atómico del nodo en SQLite
                            log_trace(
                                trace_id=str(uuid.uuid4()),
                                session_id=session_id,
                                step_order=step_order,
                                node_name=node_name,
                                duration_ms=duration_ms,
                                prompt_tokens=node_p_tokens,
                                completion_tokens=node_c_tokens,
                                tool_used=tool_name
                            )
                            step_order += 1

                        node_start = time.time()

                    # 4.4. Proceso de Evaluación y Persistencia Final
                    quality = evaluate_response_quality(user_input, final_answer)
                    verdict      = quality["verdict"]
                    score        = quality["score"]
                    reason       = quality["reason"]
                    error_msg    = reason if verdict == "bad" else None

                    # Aplicación de overrides de seguridad
                    if is_offensive or is_injection:
                        verdict   = "blocked"
                        score     = 1
                        error_msg = "Operación detenida por guardrail de seguridad."

                    # Guardado histórico de los mensajes y la evaluación
                    log_message(user_msg_id, session_id, "user", user_input,
                                is_offensive, is_injection, verdict, score, error_msg)

                    assistant_msg_id = str(uuid.uuid4())
                    log_message(assistant_msg_id, session_id, "assistant", final_answer,
                                0, 0, verdict, score, error_msg)

                    log_evaluation(
                        eval_id=str(uuid.uuid4()),
                        session_id=session_id,
                        message_id=assistant_msg_id,
                        question=user_input,
                        answer=final_answer,
                        verdict=verdict,
                        score=score,
                        reason=reason
                    )

                    # Muestra la respuesta en pantalla
                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

                except Exception as e:
                    # 4.5. Manejo de Errores de API y Filtros Nativos de Azure
                    error_str = str(e)
                    if "content_filter" in error_str or "ResponsibleAI" in error_str:
                        msg_bloqueado = "🛡️ Tu mensaje ha sido bloqueado por las políticas de seguridad del modelo subyacente."
                        st.markdown(msg_bloqueado)
                        st.session_state.messages.append({"role": "assistant", "content": msg_bloqueado})
                        log_message(str(uuid.uuid4()), session_id, "assistant", msg_bloqueado,
                                    0, 1, "blocked", 1, "Bloqueo externo (Azure Content Filter)")
                    else:
                        st.error(f"Error crítico en el flujo de ejecución: {e}")

# ==============================================================================
# 5. PANEL ANALÍTICO DE OBSERVABILIDAD (PESTAÑA 2: DASHBOARD)
# ==============================================================================
with tab_dashboard:
    st.header("Centro de Observabilidad y Telemetría")

    DB_PATH = "duocai_observability.db"

    # Verificación de existencia de datos antes de renderizar gráficos
    if not os.path.exists(DB_PATH):
        st.warning("La base de datos de observabilidad está vacía. Inicia una conversación en el chat.")
        st.stop()

    # 5.1. Conexión y Extracción de Datos (DataFrames)
    conn = sqlite3.connect(DB_PATH)
    df_msgs   = pd.read_sql_query("SELECT * FROM messages WHERE role='user'", conn)
    df_traces = pd.read_sql_query("SELECT * FROM traces", conn)
    df_evals  = pd.read_sql_query("SELECT * FROM evaluations", conn)
    conn.close()

    if df_msgs.empty:
        st.info("No hay suficientes datos. Interactúa con el chat para generar métricas.")
        st.stop()

    # 5.2. SECCIÓN: KPIs (Indicadores Clave de Rendimiento)
    st.subheader("📌 Métricas Globales de Operación")

    total_consultas   = len(df_msgs)
    lat_promedio_s    = df_traces["duration_ms"].sum() / max(total_consultas, 1) / 1000
    total_p_tokens    = df_traces["prompt_tokens"].sum()
    total_c_tokens    = df_traces["completion_tokens"].sum()
    
    # Cálculo de costo estimado en USD (Basado en pricing de gpt-4o-mini)
    costo_usd         = (total_p_tokens * 0.150 / 1_000_000) + (total_c_tokens * 0.600 / 1_000_000)

    buenos   = len(df_msgs[df_msgs["verdict"] == "good"])
    malos    = len(df_msgs[df_msgs["verdict"] == "bad"])
    bloqueados = len(df_msgs[df_msgs["verdict"] == "blocked"])
    
    tasa_error = (malos / total_consultas * 100) if total_consultas > 0 else 0
    precision  = (buenos / total_consultas * 100) if total_consultas > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("⏱ Latencia Promedio", f"{lat_promedio_s:.2f} s")
    col2.metric("📨 Interacciones", f"{total_consultas}")
    col3.metric("✅ Tasa de Precisión", f"{precision:.1f}%")
    col4.metric("❌ Tasa de Error", f"{tasa_error:.1f}%")
    col5.metric("💵 Gasto Operativo", f"${costo_usd:.5f} USD")

    st.divider()

    # 5.3. SECCIÓN: Calidad de Generación y Seguridad
    st.subheader("🛡️ Auditoría de Calidad y Filtros de Seguridad")
    col_g1, col_g2, col_g3 = st.columns(3)

    with col_g1:
        st.markdown("**Distribución de Calidad (Respuestas)**")
        verdict_data = pd.DataFrame({
            "Clasificación": ["✅ Aceptable", "❌ Deficiente", "🛡️ Interceptada"],
            "Volumen":  [buenos, malos, bloqueados]
        })
        st.bar_chart(verdict_data.set_index("Clasificación"))

    with col_g2:
        st.markdown("**Calificación Promedio (Evaluador LLM)**")
        if not df_evals.empty:
            score_por_verdict = df_evals.groupby("verdict")["score"].mean().reset_index()
            score_por_verdict.columns = ["Veredicto", "Puntaje Medio (1-10)"]
            st.bar_chart(score_por_verdict.set_index("Veredicto"))
        else:
            st.info("Esperando datos de evaluación...")

    with col_g3:
        st.markdown("**Frecuencia de Activación de Guardrails**")
        ofensivos   = int(df_msgs["is_offensive"].sum())
        inyecciones = int(df_msgs["is_prompt_injection"].sum())
        guardrail_df = pd.DataFrame({
            "Vector de Riesgo": ["🤬 Lenguaje Ofensivo", "💉 Inyección de Prompt"],
            "Detecciones": [ofensivos, inyecciones]
        })
        st.bar_chart(guardrail_df.set_index("Vector de Riesgo"))

    st.divider()

    # 5.4. SECCIÓN: Análisis Estructural (Latencia y Tokens por Nodo)
    st.subheader("⚙️ Análisis de Cuellos de Botella (Grafo)")
    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown("**Latencia Promedio por Componente (ms)**")
        node_lat = (
            df_traces.groupby("node_name")["duration_ms"]
            .mean()
            .reset_index()
            .sort_values("duration_ms", ascending=False)
        )
        node_lat.columns = ["Componente", "Tiempo Promedio (ms)"]
        st.bar_chart(node_lat.set_index("Componente"))

    with col_l2:
        st.markdown("**Consumo Estructural de Tokens**")
        token_df = df_traces.groupby("node_name")[["prompt_tokens", "completion_tokens"]].sum().reset_index()
        token_df = token_df.rename(columns={
            "node_name": "Componente",
            "prompt_tokens": "Entrada (Prompt)",
            "completion_tokens": "Salida (Completion)"
        })
        st.bar_chart(token_df.set_index("Componente"))

    st.divider()

    # 5.5. SECCIÓN: Telemetría Continua
    st.subheader("📜 Bitácora de Ejecución Histórica")

    if not df_traces.empty and "created_at" in df_traces.columns:
        df_traces["created_at"] = pd.to_datetime(df_traces["created_at"])
        latencia_tiempo = (
            df_traces.groupby(df_traces["created_at"].dt.floor("min"))["duration_ms"]
            .sum()
            .reset_index()
        )
        latencia_tiempo.columns = ["Línea de Tiempo (Min)", "Latencia Acumulada (ms)"]
        st.markdown("**Degradación de Latencia en el Tiempo**")
        st.line_chart(latencia_tiempo.set_index("Línea de Tiempo (Min)"))

    st.divider()

    # Frecuencia de uso de Herramientas
    tools_usados = df_traces[df_traces["tool_used"].notna()]
    if not tools_usados.empty:
        st.markdown("**Utilización de Herramientas (Data Sources)**")
        tool_counts = tools_usados["tool_used"].value_counts().reset_index()
        tool_counts.columns = ["Herramienta invocada", "Número de usos"]
        st.bar_chart(tool_counts.set_index("Herramienta invocada"))
        st.divider()

    # Renderizado de la tabla maestra de SQL
    st.markdown("**Log Crudo de Transacciones**")
    cols_show = [c for c in ["created_at", "session_id", "step_order", "node_name",
             "duration_ms", "prompt_tokens", "completion_tokens", "tool_used"]
             if c in df_traces.columns]
    sort_cols = [c for c in ["created_at", "step_order"] if c in df_traces.columns]
    st.dataframe(
        df_traces[cols_show].sort_values(by=sort_cols, ascending=[False, True][:len(sort_cols)]),
        use_container_width=True
    )

    st.divider()

    # 5.6. SECCIÓN: Evaluaciones Cualitativas
    st.subheader("🤖 Transcripción de Evaluaciones Autónomas")
    if not df_evals.empty:
        cols_eval = ["created_at", "verdict", "score", "reason", "question"]
        st.dataframe(
            df_evals[cols_eval].sort_values("created_at", ascending=False),
            use_container_width=True
        )
    else:
        st.info("Esperando resolución de ciclos de evaluación.")

    st.divider()

    # 5.7. SECCIÓN: Diagnóstico y Acciones Recomendadas
    st.subheader("🔍 Diagnóstico Automatizado de Sistemas")

    # Identificación del peor escenario de rendimiento
    if not df_traces.empty:
        cuello = df_traces.loc[df_traces["duration_ms"].idxmax()]
        st.warning(
            f"⚠️ **Alerta de Rendimiento:** El componente `{cuello['node_name']}` "
            f"presentó una anomalía de latencia crítica: **{cuello['duration_ms']:.0f} ms** "
            f"durante la sesión `{cuello['session_id'][:8]}...`"
        )

    # Identificación de respuestas degradadas
    if not df_evals.empty:
        bajas = df_evals[df_evals["score"] <= 4]
        if not bajas.empty:
            st.error(f"❌ Se aislaron **{len(bajas)} evento(s)** con calidad deficiente. Auditoría sugerida:")
            for _, row in bajas.iterrows():
                st.markdown(f"- **Input:** {row['question'][:80]}...  \n  **Diagnóstico del Juez:** {row['reason']}")
        else:
            st.success("✅ La integridad de las respuestas se mantiene por encima del umbral de calidad.")

    st.divider()

    st.subheader("💡 Propuestas de Mitigación (Recomendaciones de Optimización)")
    recomendaciones = []

    if lat_promedio_s > 5:
        recomendaciones.append(
            "🔴 **Mitigación de Latencia (>5s):** Implementar un modelo de Reranking. "
            "Esto permitiría mantener un 'top_k' alto (8) para calidad, pero optimizaría "
            "la entrada al LLM con solo los 3 fragmentos más relevantes."
        )

    if tasa_error > 20:
        recomendaciones.append(
            "🔴 **Mitigación de Errores (>20%):** Refinar el `QUERY_REFORMULATION_PROMPT`. "
            "Se sugiere un reajuste en el pipeline de datos aumentando el `chunk_overlap` a 400 caracteres."
        )

    if ofensivos + inyecciones > 5:
        recomendaciones.append(
            "🟡 **Ataques de Superficie:** Se registra actividad maliciosa recurrente. "
            "Se recomienda implementar un middleware de *Rate Limiting* (Limitación de Tasa) por IP o Sesión."
        )

    if tasa_error <= 10 and lat_promedio_s <= 3:
        recomendaciones.append(
            "🟢 **Estado Operacional Óptimo:** La infraestructura opera dentro de tolerancias aceptables "
            "para entornos académicos en producción."
        )

    for rec in recomendaciones:
        st.markdown(rec)