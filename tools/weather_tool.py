from fastmcp import FastMCP
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(name="weather_tool")

@mcp.tool()
def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

if __name__ == "__main__":
    mcp.run(transport="sse", port=3000)