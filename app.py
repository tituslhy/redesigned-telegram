import chainlit as cl
import logging

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatOllama(model="qwen2.5", temperature=0)

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
    cl.user_session.set("mcp_tools", {})

@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    history = cl.user_session.get("history")
    history.append(HumanMessage(content=message.content))
    logger.info(f"\n\nCurrent history: {history}\n\n")
    agent = cl.user_session.get("agent")
    logger.info("Using agent with tools")
    if agent is None:
        agent = cl.user_session.get("default_agent")
        logger.info("Using default agent")
    reply = cl.Message(content="")
    
    async for msg, metadata in agent.astream(
        {"messages": history},
        stream_mode="messages",
        config = config
    ):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            # and metadata["langgraph_node"] == "final"
        ):
            logger.info(f"\n\nMessage type: {type(msg)}")
            logger.info(f"\n\nMessage metadata: {metadata}")
            await reply.stream_token(msg.content)
    
    await reply.send()
    
@cl.on_mcp_connect
async def on_mcp_connect(connection: cl.mcp.SseMcpConnection | cl.mcp.StdioMcpConnection):
    try:
        logger.info("Connecting to MCP...")
        mcp_tools = cl.user_session.get("mcp_tools")
        if isinstance(connection.clientType, cl.mcp.StdioMcpConnection):
            logger.info("Detected Stdio MCP connection")
            connection_details = {
                "args": [connection.command],
                "transport": "stdio"
            }
        else:
            logger.info("Detected SSE MCP connection")
            connection_details = {
                "url": connection.url,
                "transport": "sse"
            }
        logger.info(f"Connection details: {connection_details}")
        
        mcp_tools[connection.name] = connection_details
        cl.user_session.set("mcp_tools", mcp_tools)
        
        mcp_client = MultiServerMCPClient(mcp_tools)
        tools = await mcp_client.get_tools()
        agent = create_react_agent(llm, tools)
        cl.user_session.set("agent", agent)
        
        logger.info(f"Connected to MCP. Connection name: {connection.name} | Connection url: {connection.url} | Connection type: {connection.clientType}")
        await cl.Message(f"Connected to MCP server: {connection.name} on {connection.url}", type="assistant_message").send()
        
    except Exception as e:
        await cl.Message(f"Error conecting to tools from MCP server: {str(e)}", type="assistant_message").send()

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str):
    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        del mcp_tools[name]
        logger.info(f"Disconnected from MCP server: {name}")
    
    if len(mcp_tools) >0:
        mcp_client = MultiServerMCPClient(mcp_tools)
        tools = await mcp_client.get_tools()
        agent = create_react_agent(llm, tools)
        cl.user_session.set("agent", agent)
    
    else:
        cl.user_session.set("agent", None)
        logger.info("No more MCP servers connected, resetting agent to None")
    
    await cl.Message(f"Disconnected from MCP server: {name}", type="assistant_message").send()