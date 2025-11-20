from src.agents.support.state import State
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from src.agents.support.nodes_support.extractor_support.prompt import SYSTEM_PROMPT

# Importamos la nueva función de utilidad desde src.utils
from src.utils import filter_messages

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")
    age: str = Field(description="The age of the person")

llm = init_chat_model("openai:gpt-4o", temperature=0)
llm = llm.with_structured_output(schema=ContactInfo)

def extractor(state: State):
    history = state["messages"]
    customer_name = state.get("customer_name", None)
    new_state: State = {}
    
    # Lógica condicional para ejecutar la extracción
    if customer_name is None or len(history) >= 10:
        
        # 1. Limpiamos el historial antes de enviarlo a OpenAI
        clean_history = filter_messages(history)
        
        # 2. Invocamos con el historial LIMPIO
        schema = llm.invoke([("system", SYSTEM_PROMPT)] + clean_history)
        
        # Actualizamos el estado con la información extraída
        new_state["customer_name"] = schema.name
        new_state["phone"] = schema.phone
        new_state["my_age"] = schema.age
        
    return new_state