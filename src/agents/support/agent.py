from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from src.agents.support.state import State
from src.agents.support.nodes_support.conversation_support.node import conversation
from src.agents.support.nodes_support.extractor_support.node import extractor
from src.agents.support.nodes_support.booking.node import booking_node
from src.agents.support.routes.intent.route import intent_route

def make_graph(config: TypedDict):
    checkpointer = config.get("checkpointer", None)
    builder = StateGraph(State)
    builder.add_node("conversation", conversation)
    builder.add_node("extractor", extractor)
    builder.add_node("booking", booking_node)

    #START
    builder.add_edge(START, 'extractor')
    builder.add_conditional_edges('extractor', intent_route)
    # END
    builder.add_edge('conversation', END)
    builder.add_edge('booking', END)

    return builder.compile(checkpointer=checkpointer)