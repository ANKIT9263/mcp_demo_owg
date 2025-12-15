from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()


def register_tools(mcp: FastMCP):
    # --------------------------------------------------
    # MATH TOOLS
    # --------------------------------------------------
    @mcp.tool()
    async def add(a: float, b: float) -> float:
        """Add two numbers"""
        return a + b

    @mcp.tool()
    async def subtract(a: float, b: float) -> float:
        """Subtract b from a"""
        return a - b

    @mcp.tool()
    async def multiply(a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b

    @mcp.tool()
    async def divide(a: float, b: float) -> float:
        """Divide a by b"""
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

    print("✅Math tools registered")
