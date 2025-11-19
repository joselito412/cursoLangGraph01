from langchain.agents import create_agent

from src.agents.support.nodes_support.booking.tools import tools
from src.agents.support.nodes_support.booking.prompt import prompt_template

booking_node = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    system_prompt=prompt_template.format(),
)

