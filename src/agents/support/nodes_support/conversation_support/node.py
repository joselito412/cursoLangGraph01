from src.agents.support.state import State
from langchain.chat_models import init_chat_model

from src.agents.support.nodes_support.conversation_support.tools import tools
from src.agents.support.nodes_support.conversation_support.prompt import SYSTEM_PROMPT

# 1. IMPORTAMOS LA UTILIDAD DE LIMPIEZA
from src.utils import filter_messages

llm = init_chat_model("openai:gpt-4o", temperature=1)
llm = llm.bind_tools(tools)

def conversation(state: State):
    history = state["messages"]
    
    # 2. LIMPIEZA: Usamos la función para dejar solo los textos válidos de la conversación.
    # Esto elimina 'file_search_call' y cualquier metadata interna que cause error 400.
    clean_history = filter_messages(history)
    
    # 3. INVOCACIÓN CON MEMORIA:
    # En lugar de enviar solo el último mensaje, enviamos 'clean_history'.
    # Así el bot recuerda lo que le dijiste hace 2 turnos (contexto).
    ai_message = llm.invoke([("system", SYSTEM_PROMPT)] + clean_history)
    
    return {"messages": [ai_message]}