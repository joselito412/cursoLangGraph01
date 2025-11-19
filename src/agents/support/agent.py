from src.agents.support.state import State
from langgraph.graph import StateGraph, START, END

from src.agents.support.nodes_support.conversation_support.node import conversation
from src.agents.support.nodes_support.extractor_support.node import extractor

builder = StateGraph(State)
builder.add_node("conversation", conversation)
builder.add_node("extractor", extractor)

builder.add_edge(START, 'extractor')
builder.add_edge('extractor', 'conversation')
builder.add_edge('conversation', END)

agent = builder.compile()