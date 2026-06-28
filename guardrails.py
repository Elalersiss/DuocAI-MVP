import os
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from prompts import (
    OFFENSIVE_GUARDRAIL_PROMPT,
    PROMPT_INJECTION_GUARDRAIL_PROMPT,
    OFF_TOPIC_GUARDRAIL_PROMPT,
)

load_dotenv()


def _evaluate(prompt: str, message: str, model: str) -> bool:
    """Llama al LLM con un prompt de guardrail y retorna True/False."""
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com",
    )
    result = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=message)])
    return result.content.strip().lower() == "true"


def run_guardrails(message: str, model: str = "gpt-4o-mini") -> dict:
    """
    Ejecuta los 3 evaluadores de seguridad EN PARALELO para minimizar latencia.

    Retorna:
        {
            "is_offensive":        bool,
            "is_prompt_injection": bool,
            "is_off_topic":        bool,
        }
    """
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_offensive  = executor.submit(_evaluate, OFFENSIVE_GUARDRAIL_PROMPT,        message, model)
        future_injection  = executor.submit(_evaluate, PROMPT_INJECTION_GUARDRAIL_PROMPT, message, model)
        future_off_topic  = executor.submit(_evaluate, OFF_TOPIC_GUARDRAIL_PROMPT,        message, model)

    return {
        "is_offensive":        future_offensive.result(),
        "is_prompt_injection": future_injection.result(),
        "is_off_topic":        future_off_topic.result(),
    }