import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from tools import search_knowledge_base, get_order_status, check_return_eligibility, create_support_ticket



load_dotenv()

agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[search_knowledge_base, get_order_status, check_return_eligibility, create_support_ticket],
    checkpointer=InMemorySaver(),
)

def run_agent(question: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    return result["messages"][-1].content