"""
Generic + Math Tools for MCP
----------------------------
Supports conversational tools and mathematical operations.
"""

import os

from fastmcp import FastMCP
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()


def register_tools(mcp: FastMCP):
    OPENAI_MODEL = os.getenv("OPENAI_MODEL")

    # --------------------------------------------------
    # CONVERSATIONAL TOOLS
    # --------------------------------------------------
    @mcp.tool()
    async def handle_greeting(text: str, openai_api_key: str) -> str:
        if any(g in text.lower() for g in ["hi", "hello", "good morning"]):
            llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.6, api_key=openai_api_key)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Reply warmly to greetings."),
                    ("user", f"User said: {text}"),
                ]
            )
            return (await (prompt | llm | StrOutputParser()).ainvoke({})).strip()
        return "No greeting detected."

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

    print("✅ Generic + Math tools registered")
