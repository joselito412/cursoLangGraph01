# ✅ Línea 1: Correcta (ya tiene src)
from src.agents.support.state import State

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# ✅ Línea 4: CORREGIDA (agregamos src. al inicio)
from src.agents.support.nodes_support.extractor_support.prompt import SYSTEM_PROMPT

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
    # Nota: Asegúrate que la lógica de "my_age" coincida con tu State
    if customer_name is None or len(history) >= 10:
        schema = llm.invoke([("system", SYSTEM_PROMPT)] + history)
        new_state["customer_name"] = schema.name
        new_state["phone"] = schema.phone
        new_state["my_age"] = schema.age
    return new_state