import os
from dotenv import load_dotenv
from langsmith import traceable

# ==============================================================================
# 1. CONFIGURACIÓN DEL ENTORNO Y VARIABLES DE ACCESO
# ==============================================================================
# Carga de credenciales locales y aprovisionamiento de variables del sistema
load_dotenv()

# Configuración del motor de inferencia remota (GitHub Models / Azure AI)
token = os.getenv("GITHUB_TOKEN")
os.environ["OPENAI_API_KEY"] = token
os.environ["OPENAI_API_BASE"] = "https://models.inference.ai.azure.com"

# Inicialización de las variables globales de telemetría para LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "DuocAI_Proyecto")
os.environ["LANGSMITH_TRACING"] = "true"

# ==============================================================================
# 2. CONFIGURACIÓN DE TELEMETRÍA Y CALLBACKS DE RESPALDO
# ==============================================================================
import litellm

# Enrutamiento de logs internos de inferencia hacia la plataforma de observabilidad
litellm.success_callback = ["langsmith"]
litellm.failure_callback = ["langsmith"]

# Inyección de metadatos de sesión para el seguimiento de hilos de conversación
litellm.metadata = {
    "thread_id": "sesion_prueba_alexis_2026"
}

# ==============================================================================
# 3. IMPORTACIÓN DE COMPONENTES DEL FRAMEWORK Y CONFIGURACIÓN DEL LLM
# ==============================================================================
from crewai import Agent, Task, Crew, Process, LLM 
from tools import herramienta_duoc, herramienta_calendario

# Configuración y parametrización del Modelo de Lenguaje de Gran Escala (LLM)
estudiante_llm = LLM(
    model="gpt-4o-mini",
    api_key=token,
    base_url="https://models.inference.ai.azure.com",
    temperature=0
)

# ==============================================================================
# 4. DEFINICIÓN DE PERFILES COGNITIVOS (AGENTES AUTÓNOMOS)
# ==============================================================================
# Agente encargado de la extracción fáctica y auditoría de documentos fuente
investigador_normativo = Agent(
    role='Investigador Normativo',
    goal='Extraer información exacta de los reglamentos y fechas sobre {topic}',
    backstory='Eres un auditor experto en documentos y plazos institucionales de Duoc UC.',
    tools=[herramienta_duoc, herramienta_calendario],
    llm=estudiante_llm,
    verbose=True,
    allow_delegation=False
)

# Agente encargado de la síntesis adaptativa y el alineamiento del tono estudiantil
asesor_estudiantil = Agent(
    role='Asesor de Apoyo al Estudiante',
    goal='Explicar los reglamentos de forma clara al alumno sobre {topic}',
    backstory='Eres un consejero que traduce normas técnicas a lenguaje amigable.',
    llm=estudiante_llm,
    verbose=True,
    allow_delegation=False
)

# ==============================================================================
# 5. PIPELINE DE EJECUCIÓN Y ORQUESTACIÓN JERÁRQUICA (CREW)
# ==============================================================================
@traceable(run_type="chain", name="Flujo_DuocAI_Crew")
def ejecutar_agente_duoc(pregunta_usuario):
    """
    Orquesta el ciclo de vida del proceso multi-agente jerárquico.
    
    Toma la entrada del usuario, define las subtareas de investigación y asesoría,
    y ejecuta la simulación controlada por el LLM supervisor (manager_llm).
    Captura de forma síncrona el rastro completo en el árbol raíz de LangSmith.
    """
    # Definición de la subtarea de auditoría documental y extracción
    tarea_investigacion = Task(
        description='Busca en los reglamentos o fechas todo lo relacionado con: {topic}',
        expected_output='Un informe técnico con los fragmentos de la norma o fechas exactas.',
        agent=investigador_normativo
    )

    # Definición de la subtarea de procesamiento semántico y formateo empático
    tarea_asesoria = Task(
        description='Usa el informe técnico para responder al alumno sobre: {topic}',
        expected_output='Una respuesta estructurada y amigable para el estudiante.',
        agent=asesor_estudiantil,
        context=[tarea_investigacion]
    )

    # Consolidación de la Crew bajo un modelo de toma de decisiones jerárquico
    equipo_duoc = Crew(
        agents=[investigador_normativo, asesor_estudiantil],
        tasks=[tarea_investigacion, tarea_asesoria],
        process=Process.hierarchical, 
        manager_llm=estudiante_llm,
        verbose=True,
        memory=False 
    )

    # Disparo e inicio de la cadena de razonamiento agéntico (Chain-of-Thought)
    resultado = equipo_duoc.kickoff(inputs={'topic': pregunta_usuario})
    return str(resultado)