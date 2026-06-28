import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import uuid
import time
import sqlite3
import pandas as pd
import os
import json

from agent import app as agent_graph
from observability import log_session, log_message, log_trace, log_evaluation

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
st.set_page_config(
    page_title="DuocAI - Observabilidad EV3",
    page_icon="🤖",
    layout="wide"
)
st.title("DuocAI: Asistente Académico con Observabilidad")

tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard de Observabilidad (EV3)"])

# ==============================================================================
# VARIABLES DE SESIÓN
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ==============================================================================
# JUEZ LLM — Agent Goal Accuracy
# Evalúa si la respuesta final fue realmente útil para el usuario.
# Inspirado en el test_eval.py del profesor (clase 3.1).
# ==============================================================================
def evaluate_response_quality(question: str, answer: str) -> dict:
    """
    Llama a un LLM como juez para determinar si la respuesta fue buena.
    Retorna: {"verdict": "good"|"bad", "score": 1-10, "reason": str}
    """
    # Si la respuesta está vacía o es el mensaje de bloqueo, no evaluamos con LLM
    if not answer or "bloqueado por las políticas" in answer:
        return {"verdict": "blocked", "score": 1, "reason": "Mensaje bloqueado por guardrails."}

    try:
        judge_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url="https://models.inference.ai.azure.com",
        )
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
        # Limpiamos posibles backticks de markdown antes de parsear
        raw = result.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        # Si el juez falla, usamos heurística simple como fallback
        bad_signals = ["no se encontró", "lamentablemente", "no tengo información",
                       "no puedo responder", "error en la base de datos"]
        is_bad = any(s in answer.lower() for s in bad_signals)
        return {
            "verdict": "bad" if is_bad else "good",
            "score": 3 if is_bad else 8,
            "reason": "Evaluación heurística (juez LLM no disponible)."
        }


# ==============================================================================
# PESTAÑA 1: CHAT
# ==============================================================================
with tab_chat:
    st.markdown("Bienvenido. Este sistema utiliza LangGraph y RAG sobre normativas de Duoc UC.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Escribe tu consulta aquí..."):
        session_id = st.session_state.thread_id
        log_session(session_id)

        user_msg_id = str(uuid.uuid4())

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Ejecutando grafo de agentes..."):
                try:
                    config = {"configurable": {"thread_id": session_id}}

                    final_answer = ""
                    is_offensive  = 0
                    is_injection  = 0
                    step_order    = 0

                    node_start = time.time()

                    # Ejecutamos el grafo en modo stream para capturar cada nodo
                    for chunk in agent_graph.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                        stream_mode="updates"
                    ):
                        node_end  = time.time()
                        duration_ms = (node_end - node_start) * 1000

                        for node_name, state in chunk.items():
                            tool_name   = None
                            # CORRECCIÓN: los tokens se leen solo del mensaje de ESTE nodo,
                            # no del acumulado global, evitando el doble conteo.
                            node_p_tokens = 0
                            node_c_tokens = 0
                            messages    = state.get("messages", [])

                            # Capturamos flags de guardrails
                            if node_name == "check_guardrails":
                                is_offensive = 1 if state.get("is_offensive") else 0
                                is_injection = 1 if state.get("is_prompt_injection") else 0

                            for msg in messages:
                                if isinstance(msg, AIMessage) and msg.tool_calls:
                                    tool_name = msg.tool_calls[0]["name"]
                                if isinstance(msg, AIMessage) and msg.usage_metadata:
                                    node_p_tokens = msg.usage_metadata.get("input_tokens", 0)
                                    node_c_tokens = msg.usage_metadata.get("output_tokens", 0)
                                if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                                    final_answer = msg.content

                            # Guardamos la traza de este nodo con su step_order
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

                    # --- Evaluación de calidad con juez LLM ---
                    quality = evaluate_response_quality(user_input, final_answer)
                    verdict      = quality["verdict"]
                    score        = quality["score"]
                    reason       = quality["reason"]
                    error_msg    = reason if verdict == "bad" else None

                    # Sobreescribimos con "blocked" si los guardrails actuaron
                    if is_offensive or is_injection:
                        verdict   = "blocked"
                        score     = 1
                        error_msg = "Mensaje bloqueado por guardrail de seguridad."

                    # Persistimos mensajes
                    log_message(user_msg_id, session_id, "user", user_input,
                                is_offensive, is_injection, verdict, score, error_msg)

                    assistant_msg_id = str(uuid.uuid4())
                    log_message(assistant_msg_id, session_id, "assistant", final_answer,
                                0, 0, verdict, score, error_msg)

                    # Persistimos evaluación LLM
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

                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

                except Exception as e:
                    error_str = str(e)
                    if "content_filter" in error_str or "ResponsibleAI" in error_str:
                        msg_bloqueado = "🛡️ Tu mensaje ha sido bloqueado por las políticas de seguridad de DuocAI."
                        st.markdown(msg_bloqueado)
                        st.session_state.messages.append({"role": "assistant", "content": msg_bloqueado})
                        log_message(str(uuid.uuid4()), session_id, "assistant", msg_bloqueado,
                                    0, 1, "blocked", 1, "Bloqueado por filtro de contenido de Azure")
                    else:
                        st.error(f"Error crítico en el flujo del agente: {e}")


# ==============================================================================
# PESTAÑA 2: DASHBOARD DE OBSERVABILIDAD
# ==============================================================================
with tab_dashboard:
    st.header("Centro de Observabilidad — EV3 DuocAI")

    DB_PATH = "duocai_observability.db"

    if not os.path.exists(DB_PATH):
        st.warning("Aún no hay datos. Realiza al menos una consulta en el chat.")
        st.stop()

    conn = sqlite3.connect(DB_PATH)
    df_msgs   = pd.read_sql_query("SELECT * FROM messages WHERE role='user'", conn)
    df_traces = pd.read_sql_query("SELECT * FROM traces", conn)
    df_evals  = pd.read_sql_query("SELECT * FROM evaluations", conn)
    conn.close()

    if df_msgs.empty:
        st.info("Realiza algunas consultas en el chat para poblar los gráficos.")
        st.stop()

    # ------------------------------------------------------------------
    # SECCIÓN 1 — KPIs principales
    # ------------------------------------------------------------------
    st.subheader("📌 Métricas Clave de Rendimiento (KPIs)")

    total_consultas   = len(df_msgs)
    lat_promedio_s    = df_traces["duration_ms"].sum() / max(total_consultas, 1) / 1000
    total_p_tokens    = df_traces["prompt_tokens"].sum()
    total_c_tokens    = df_traces["completion_tokens"].sum()
    costo_usd         = (total_p_tokens * 0.150 / 1_000_000) + (total_c_tokens * 0.600 / 1_000_000)

    buenos   = len(df_msgs[df_msgs["verdict"] == "good"])
    malos    = len(df_msgs[df_msgs["verdict"] == "bad"])
    bloqueados = len(df_msgs[df_msgs["verdict"] == "blocked"])
    tasa_error = (malos / total_consultas * 100) if total_consultas > 0 else 0
    precision  = (buenos / total_consultas * 100) if total_consultas > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("⏱ Latencia Promedio", f"{lat_promedio_s:.2f} s")
    col2.metric("📨 Consultas Totales", f"{total_consultas}")
    col3.metric("✅ Precisión", f"{precision:.1f}%")
    col4.metric("❌ Tasa de Error", f"{tasa_error:.1f}%")
    col5.metric("💵 Costo Acumulado", f"${costo_usd:.5f} USD")

    st.divider()

    # ------------------------------------------------------------------
    # SECCIÓN 2 — Calidad y Guardrails
    # ------------------------------------------------------------------
    st.subheader("🛡️ Calidad de Respuestas y Seguridad")
    col_g1, col_g2, col_g3 = st.columns(3)

    with col_g1:
        st.markdown("**Distribución de Calidad**")
        # Construimos el DataFrame con las 3 categorías siempre presentes
        verdict_data = pd.DataFrame({
            "Resultado": ["✅ Buena", "❌ Mala", "🛡️ Bloqueada"],
            "Cantidad":  [buenos, malos, bloqueados]
        })
        st.bar_chart(verdict_data.set_index("Resultado"))

    with col_g2:
        st.markdown("**Score Promedio de Calidad (LLM Judge)**")
        if not df_evals.empty:
            score_por_verdict = df_evals.groupby("verdict")["score"].mean().reset_index()
            score_por_verdict.columns = ["Verdict", "Score promedio"]
            st.bar_chart(score_por_verdict.set_index("Verdict"))
        else:
            st.info("Sin evaluaciones LLM aún.")

    with col_g3:
        st.markdown("**Activaciones de Guardrails**")
        ofensivos   = int(df_msgs["is_offensive"].sum())
        inyecciones = int(df_msgs["is_prompt_injection"].sum())
        guardrail_df = pd.DataFrame({
            "Guardrail": ["🤬 Ofensivo", "💉 Prompt Injection"],
            "Activaciones": [ofensivos, inyecciones]
        })
        st.bar_chart(guardrail_df.set_index("Guardrail"))

    st.divider()

    # ------------------------------------------------------------------
    # SECCIÓN 3 — Latencia y Recursos
    # ------------------------------------------------------------------
    st.subheader("⚙️ Rendimiento del Grafo de Agentes")
    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown("**Latencia Promedio por Nodo (ms)**")
        node_lat = (
            df_traces.groupby("node_name")["duration_ms"]
            .mean()
            .reset_index()
            .sort_values("duration_ms", ascending=False)
        )
        node_lat.columns = ["Nodo", "Latencia Promedio (ms)"]
        st.bar_chart(node_lat.set_index("Nodo"))

    with col_l2:
        st.markdown("**Uso de Tokens por Nodo**")
        token_df = df_traces.groupby("node_name")[["prompt_tokens", "completion_tokens"]].sum().reset_index()
        token_df = token_df.rename(columns={
            "node_name": "Nodo",
            "prompt_tokens": "Prompt",
            "completion_tokens": "Completion"
        })
        st.bar_chart(token_df.set_index("Nodo"))

    st.divider()

    # ------------------------------------------------------------------
    # SECCIÓN 4 — Trazabilidad Histórica (IE3 / IE4)
    # ------------------------------------------------------------------
    st.subheader("📜 Trazabilidad Histórica de Ejecución")

    # Evolución de latencia total por consulta a lo largo del tiempo
    if not df_traces.empty and "created_at" in df_traces.columns:
        df_traces["created_at"] = pd.to_datetime(df_traces["created_at"])
        latencia_tiempo = (
            df_traces.groupby(df_traces["created_at"].dt.floor("min"))["duration_ms"]
            .sum()
            .reset_index()
        )
        latencia_tiempo.columns = ["Minuto", "Latencia Total (ms)"]
        st.markdown("**Evolución de Latencia Total en el Tiempo**")
        st.line_chart(latencia_tiempo.set_index("Minuto"))

    st.divider()

    # Tools más usadas
    tools_usados = df_traces[df_traces["tool_used"].notna()]
    if not tools_usados.empty:
        st.markdown("**Frecuencia de Uso de Tools**")
        tool_counts = tools_usados["tool_used"].value_counts().reset_index()
        tool_counts.columns = ["Tool", "Veces usada"]
        st.bar_chart(tool_counts.set_index("Tool"))
        st.divider()

    # Tabla completa de trazas
    st.markdown("**Registro Detallado de Nodos Ejecutados**")
    cols_show = [c for c in ["created_at", "session_id", "step_order", "node_name",
             "duration_ms", "prompt_tokens", "completion_tokens", "tool_used"]
             if c in df_traces.columns]
    sort_cols = [c for c in ["created_at", "step_order"] if c in df_traces.columns]
    st.dataframe(
        df_traces[cols_show].sort_values(by=sort_cols, ascending=[False, True][:len(sort_cols)]),
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------------------
    # SECCIÓN 5 — Últimas evaluaciones LLM (IE1 / IE4)
    # ------------------------------------------------------------------
    st.subheader("🤖 Evaluaciones de Calidad (Agent Goal Accuracy — Juez LLM)")
    if not df_evals.empty:
        cols_eval = ["created_at", "verdict", "score", "reason", "question"]
        st.dataframe(
            df_evals[cols_eval].sort_values("created_at", ascending=False),
            use_container_width=True
        )
    else:
        st.info("Aún no hay evaluaciones LLM registradas.")

    st.divider()

    # ------------------------------------------------------------------
    # SECCIÓN 6 — Análisis de Patrones y Anomalías (IE4 / IE7)
    # ------------------------------------------------------------------
    st.subheader("🔍 Análisis de Patrones y Anomalías")

    # Nodo más lento (cuello de botella)
    if not df_traces.empty:
        cuello = df_traces.loc[df_traces["duration_ms"].idxmax()]
        st.warning(
            f"⚠️ **Cuello de botella detectado:** El nodo `{cuello['node_name']}` "
            f"registró la mayor latencia individual: **{cuello['duration_ms']:.0f} ms** "
            f"en la sesión `{cuello['session_id'][:8]}...`"
        )

    # Conversaciones con score bajo
    if not df_evals.empty:
        bajas = df_evals[df_evals["score"] <= 4]
        if not bajas.empty:
            st.error(f"❌ **{len(bajas)} respuesta(s) con score ≤ 4** detectadas. Revisar:")
            for _, row in bajas.iterrows():
                st.markdown(f"- **Pregunta:** {row['question'][:80]}...  \n  **Razón:** {row['reason']}")
        else:
            st.success("✅ No se detectaron respuestas de baja calidad.")

    st.divider()

    # ------------------------------------------------------------------
    # SECCIÓN 7 — Recomendaciones de Optimización (IE7)
    # ------------------------------------------------------------------
    st.subheader("💡 Recomendaciones de Optimización del Sistema")

    recomendaciones = []

    if lat_promedio_s > 5:
        recomendaciones.append(
            "🔴 **Alta latencia promedio (>5s):** Considerar reducir `top_k` en el RAG "
            "de 5 a 3 fragmentos, o usar un modelo más pequeño para los guardrails."
        )

    if tasa_error > 20:
        recomendaciones.append(
            "🔴 **Alta tasa de error (>20%):** Revisar el `QUERY_REFORMULATION_PROMPT` "
            "y el tamaño de los chunks en la ingesta (actualmente 1200 chars). "
            "Aumentar el `chunk_overlap` de 300 a 400 podría mejorar la recuperación."
        )

    if ofensivos + inyecciones > 5:
        recomendaciones.append(
            "🟡 **Múltiples activaciones de guardrails:** El sistema está siendo testeado "
            "con mensajes maliciosos. Considerar añadir rate limiting por sesión."
        )

    if tasa_error <= 10 and lat_promedio_s <= 3:
        recomendaciones.append(
            "🟢 **Sistema operando dentro de parámetros óptimos.** "
            "Latencia y precisión en rangos aceptables para producción académica."
        )

    for rec in recomendaciones:
        st.markdown(rec)