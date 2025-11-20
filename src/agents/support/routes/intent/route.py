from pydantic import BaseModel, Field
from typing import Literal
from langchain.chat_models import init_chat_model
from src.agents.support.state import State
from src.agents.support.routes.intent.prompt import SYSTEM_PROMPT
from src.utils import filter_messages

class RouteIntent(BaseModel):
    step: Literal["conversation", "booking"] = Field(
        'conversation', description="The next step in the routing process"
    )

llm = init_chat_model("openai:gpt-4o", temperature=0)
llm = llm.with_structured_output(schema=RouteIntent)

def intent_route(state: State) -> Literal["conversation", "booking"]:
    history = state["messages"]
    
    # Limpiamos el historial usando la utilidad centralizada
    clean_history = filter_messages(history)

    # Invocamos al LLM con el historial limpio
    schema = llm.invoke([("system", SYSTEM_PROMPT)] + clean_history)
    
    if schema.step is not None:
        return schema.step
        
    return 'conversation'