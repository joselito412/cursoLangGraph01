# ✅ 1. CORREGIDO: Agregamos 'src.' al inicio
from src.agents.support.state import State
from langchain.chat_models import init_chat_model

# ✅ 2. CORREGIDO: Agregamos 'src.' a estas dos líneas también
from src.agents.support.nodes_support.conversation_support.tools import tools
from src.agents.support.nodes_support.conversation_support.prompt import SYSTEM_PROMPT

llm = init_chat_model("openai:gpt-4o", temperature=1)

# ⚠️ NOTA: Si tu variable 'tools' ya es una lista [tool1, tool2], 
# no necesitas poner corchetes extra. Lo he dejado como bind_tools(tools).
# Si te da error, prueba poner de nuevo los corchetes: bind_tools([tools])
llm = llm.bind_tools(tools)

def conversation(state: State):
    new_state: State = {}
    history = state["messages"]
    
    # Obtenemos el último mensaje
    last_message = history[-1]
    
    # Esta variable la obtienes pero no la estabas usando en el prompt. 
    # (La dejo por si planeas usarla luego).
    customer_name = state.get("customer_name", 'John Doe')

    # Los mensajes de LangChain usan '.content', NO '.text'.
    ai_message = llm.invoke([
        ("system", SYSTEM_PROMPT), 
        ("user", last_message.content) 
    ])
    
    new_state["messages"] = [ai_message]
    return new_state