import sqlite3
import os
import json
from datetime import datetime

DB_PATH = "duocai_observability.db"

def init_db():
    """Inicializa las tablas de observabilidad histórica para la EV3."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Tabla de Sesiones
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT
        )
    """)

    # 2. Tabla de Mensajes
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

    # 3. Tabla de Trazas — ahora incluye step_order para reconstruir el flujo completo
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

    # 4. Tabla de Evaluaciones LLM (Agent Goal Accuracy)
    #    Registra si la respuesta final fue útil según un juez LLM
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            eval_id     TEXT PRIMARY KEY,
            session_id  TEXT,
            message_id  TEXT,
            question    TEXT,
            answer      TEXT,
            verdict     TEXT,   -- 'good' | 'bad' | 'blocked'
            score       INTEGER,
            reason      TEXT,
            created_at  TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
        (session_id, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def log_message(message_id: str, session_id: str, role: str, content: str,
                is_offensive: int = 0, is_prompt_injection: int = 0,
                verdict: str = "good", score: int = 10, error_msg: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages
            (message_id, session_id, role, content, is_offensive,
             is_prompt_injection, verdict, score, error_msg, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (message_id, session_id, role, content, is_offensive,
          is_prompt_injection, verdict, score, error_msg,
          datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def log_trace(trace_id: str, session_id: str, node_name: str,
              duration_ms: float, step_order: int = 0,
              prompt_tokens: int = 0, completion_tokens: int = 0,
              tool_used: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO traces
            (trace_id, session_id, step_order, node_name, duration_ms,
             prompt_tokens, completion_tokens, tool_used, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (trace_id, session_id, step_order, node_name, duration_ms,
          prompt_tokens, completion_tokens, tool_used,
          datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def log_evaluation(eval_id: str, session_id: str, message_id: str,
                   question: str, answer: str,
                   verdict: str, score: int, reason: str):
    """Persiste el resultado del juez LLM (Agent Goal Accuracy)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evaluations
            (eval_id, session_id, message_id, question, answer,
             verdict, score, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (eval_id, session_id, message_id, question, answer,
          verdict, score, reason, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


# Inicializamos al importar
init_db()