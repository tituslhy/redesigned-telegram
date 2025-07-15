import chainlit as cl
import logging

from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatOllama(model="llama3.1", temperature=0)

def setup_default() -> StateGraph:
    def invoke_llm(state: MessagesState):
        return {"messages": [llm.invoke(state['messages'])]}
    builder = StateGraph(MessagesState)
    builder.add_node("invoke_llm", invoke_llm)
    builder.add_edge(START, "invoke_llm")
    builder.add_edge("invoke_llm", END)
    graph = builder.compile()
    return graph

@cl.on_chat_start
async def on_chat_start():
    default_agent = setup_default()
    cl.user_session.set("default_agent", default_agent)
    cl.user_session.set("agent", None)
    cl.user_session.set("history", [])

@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    history = cl.user_session.get("history")
    history.append(HumanMessage(content=message.content))
    agent = cl.user_session.get("agent")
    if agent is None:
        agent = cl.user_session.get("default_agent")
    
    reply = cl.Message(content="")
    
    for msg, metadata in agent.stream(
        {"messages": history},
        stream_mode="messages",
        config = config
    ):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            # and metadata["langgraph_node"] == "final"
        ):
            await reply.stream_token(msg.content)
    
    await reply.send()
    
    