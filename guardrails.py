"""
=============================================================================
MÓDULO DE SEGURIDAD Y FILTRADO DE CONTENIDO (GUARDRAILS)
=============================================================================
Este archivo implementa la capa de seguridad periférica del asistente.
Para evitar ataques, malas prácticas o consultas fuera de foco, cada mensaje
es evaluado por tres clasificadores LLM independientes que corren de forma 
asíncrona y paralela, garantizando una latencia mínima en la respuesta.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Importación de las plantillas de instrucciones (prompts) de los evaluadores
from prompts import (
    OFFENSIVE_GUARDRAIL_PROMPT,
    PROMPT_INJECTION_GUARDRAIL_PROMPT,
    OFF_TOPIC_GUARDRAIL_PROMPT,
)

# Inicialización de las variables de entorno locales
load_dotenv()

# ==============================================================================
# 1. FUNCIÓN INTERNA DE EVALUACIÓN (LLM CLASSIFIER)
# ==============================================================================
def _evaluate(prompt: str, message: str, model: str) -> bool:
    """
    Función interna de soporte que realiza una consulta binaria a la API.
    Envía el mensaje del usuario junto con las instrucciones específicas del
    filtro de seguridad para obtener un veredicto estricto (true o false).

    Args:
        prompt (str): Las instrucciones de moderación o reglas que el modelo debe seguir.
        message (str): El texto exacto ingresado por el usuario en el chat.
        model (str): El identificador del modelo de lenguaje a utilizar.

    Returns:
        bool: True si el mensaje viola la regla de seguridad del prompt; 
              False en caso de que el mensaje sea legítimo y seguro.
    """
    # Conexión directa a la pasarela de inferencia asignada
    llm = ChatOpenAI(
        model=model,
        temperature=0,  # Temperatura 0 para asegurar respuestas deterministas sin variaciones
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com",
    )
    
    # Se ejecuta la llamada enviando la regla (System) y el input a evaluar (Human)
    result = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=message)])
    
    # Limpieza de espacios y normalización a minúsculas del resultado textual para la conversión a Booleano
    return result.content.strip().lower() == "true"


# ==============================================================================
# 2. ORQUESTACIÓN DE EVALUADORES EN PARALELO
# ==============================================================================
def run_guardrails(message: str, model: str = "gpt-4o-mini") -> dict:
    """
    Ejecuta de forma simultánea los tres evaluadores de seguridad utilizando hilos.
    Esta aproximación técnica evita el cuello de botella secuencial (hacer tres llamadas 
    una tras otra), logrando que la verificación de seguridad completa tome 
    únicamente el tiempo de la llamada individual más lenta.

    Args:
        message (str): El texto del usuario que se va a escanear preventivamente.
        model (str, opcional): El modelo de inferencia por defecto. Registrado como 'gpt-4o-mini'.

    Returns:
        dict: Un diccionario con el mapa completo de riesgos detectados:
              {
                  "is_offensive": bool,        # Detección de lenguaje vulgar o insultos
                  "is_prompt_injection": bool, # Intentos de manipulación del sistema o prompts
                  "is_off_topic": bool         # Desviación temática fuera del entorno institucional
              }
    """
    # Se inicializa el administrador de hilos asignando un máximo de 3 trabajadores en paralelo
    with ThreadPoolExecutor(max_workers=3) as executor:
        
        # Despacho asíncrono del Evaluador 1: Contenido Ofensivo
        future_offensive  = executor.submit(_evaluate, OFFENSIVE_GUARDRAIL_PROMPT,        message, model)
        
        # Despacho asíncrono del Evaluador 2: Prompt Injection
        future_injection  = executor.submit(_evaluate, PROMPT_INJECTION_GUARDRAIL_PROMPT, message, model)
        
        # Despacho asíncrono del Evaluador 3: Temas fuera de dominio (Off-Topic)
        future_off_topic  = executor.submit(_evaluate, OFF_TOPIC_GUARDRAIL_PROMPT,        message, model)

    # El bloque 'with' finaliza de forma segura una vez que los tres hilos resuelven, 
    # consolidando y retornando las respuestas recopiladas en un formato JSON estructurado
    return {
        "is_offensive":        future_offensive.result(),
        "is_prompt_injection": future_injection.result(),
        "is_off_topic":        future_off_topic.result(),
    }