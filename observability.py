"""
=============================================================================
MÓDULO DE PERSISTENCIA Y OBSERVABILIDAD (SQLITE)
=============================================================================
Este archivo gestiona la telemetría y el almacenamiento histórico del sistema.
Utiliza SQLite para persistir el ciclo de vida de cada interacción, desde la 
creación de la sesión hasta la ejecución milimétrica de cada nodo del grafo, 
permitiendo un análisis de rendimiento, costos y calidad de respuestas.
"""

import sqlite3
import os
import json
from datetime import datetime

# Ruta local donde se almacenará y persistirá la base de datos relacional
DB_PATH = "duocai_observability.db"

# ==============================================================================
# 1. INICIALIZACIÓN DE LA BASE DE DATOS Y ESQUEMAS (DDL)
# ==============================================================================
def init_db():
    """
    Inicializa las tablas de observabilidad histórica.
    Garantiza que la estructura relacional exista antes de intentar registrar datos.
    Se compone de 4 tablas principales: sessions, messages, traces y evaluations.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1.1 Tabla de Sesiones: Controla los hilos de conversación únicos.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT
        )
    """)

    # 1.2 Tabla de Mensajes: Almacena el texto crudo y banderas de seguridad preventivas.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id   TEXT PRIMARY KEY,
            session_id   TEXT,
            role         TEXT,
            content      TEXT,
            is_offensive INTEGER DEFAULT 0,
            is_prompt_injection INTEGER DEFAULT 0,
            verdict      TEXT DEFAULT 'good',
            score        INTEGER DEFAULT 10,
            error_msg    TEXT,
            created_at   TEXT
        )
    """)

    # 1.3 Tabla de Trazas: Registra la ejecución atómica de los nodos de LangGraph.
    # El campo 'step_order' es crítico para reconstruir cronológicamente el grafo.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            trace_id          TEXT PRIMARY KEY,
            session_id        TEXT,
            step_order        INTEGER DEFAULT 0,
            node_name         TEXT,
            duration_ms       REAL,
            prompt_tokens     INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            tool_used         TEXT,
            created_at        TEXT
        )
    """)

    # 1.4 Tabla de Evaluaciones LLM (Agent Goal Accuracy): 
    # Almacena la auditoría cualitativa realizada por el Juez LLM autónomo.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            eval_id     TEXT PRIMARY KEY,
            session_id  TEXT,
            message_id  TEXT,
            question    TEXT,
            answer      TEXT,
            verdict     TEXT,   -- Admite valores: 'good', 'bad', 'blocked'
            score       INTEGER,
            reason      TEXT,
            created_at  TEXT
        )
    """)

    conn.commit()
    conn.close()


# ==============================================================================
# 2. FUNCIONES DE REGISTRO TRANSACCIONAL (DML)
# ==============================================================================

def log_session(session_id: str):
    """
    Registra una nueva sesión en el sistema. 
    Usa 'INSERT OR IGNORE' para evitar duplicados si la sesión ya existe.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
        (session_id, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def log_message(message_id: str, session_id: str, role: str, content: str,
                is_offensive: int = 0, is_prompt_injection: int = 0,
                verdict: str = "good", score: int = 10, error_msg: str = None):
    """
    Persiste un mensaje individual (del usuario o del asistente) asociándolo
    a las métricas de contingencia y bloqueos de seguridad.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages
            (message_id, session_id, role, content, is_offensive,
             is_prompt_injection, verdict, score, error_msg, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (message_id, session_id, role, content, is_offensive,
          is_prompt_injection, verdict, score, error_msg,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()


def log_trace(trace_id: str, session_id: str, node_name: str,
              duration_ms: float, step_order: int = 0,
              prompt_tokens: int = 0, completion_tokens: int = 0,
              tool_used: str = None):
    """
    Registra el costo computacional y temporal de un nodo específico de LangGraph.
    Fundamental para el análisis de cuellos de botella y cálculo de costos por API.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO traces
            (trace_id, session_id, step_order, node_name, duration_ms,
             prompt_tokens, completion_tokens, tool_used, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (trace_id, session_id, step_order, node_name, duration_ms,
          prompt_tokens, completion_tokens, tool_used,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()


def log_evaluation(eval_id: str, session_id: str, message_id: str,
                   question: str, answer: str,
                   verdict: str, score: int, reason: str):
    """
    Persiste el resultado de la auditoría de calidad generada por el Juez LLM.
    Almacena la justificación semántica ('reason') para permitir revisión humana posterior.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evaluations
            (eval_id, session_id, message_id, question, answer,
             verdict, score, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (eval_id, session_id, message_id, question, answer,
          verdict, score, reason, datetime.now().isoformat()))
    conn.commit()
    conn.close()


# ==============================================================================
# 3. EJECUCIÓN AUTOMÁTICA
# ==============================================================================
# Garantiza que el esquema SQL esté construido en el momento en que 
# cualquier otro archivo importe este módulo.
init_db()