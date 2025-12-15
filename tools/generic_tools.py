
import os

from fastmcp import FastMCP
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

    print("✅ Generic tools registered")
