import os
import json
import asyncio
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastmcp.client import Client

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

MCP_ENDPOINT = "http://localhost:8080/mcp"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# --------------------------------------------------
# LANGCHAIN-SAFE MULTI-STEP PLANNER PROMPT
# --------------------------------------------------
PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a planner.\n"
                "Break the user query into ordered tool steps.\n\n"
                "Rules:\n"
                "- Return ONLY valid JSON\n"
                "- Output must be a JSON list\n"
                "- Each item must contain: tool, args\n"
                "- If a step depends on the previous result, "
                "use the literal string PREVIOUS_RESULT as a value\n"
            ),
        ),
        (
            "user",
            (
                "Available tools:\n{tools}\n\n"
                "User query:\n{query}\n\n"
                "Return the execution steps now."
            ),
        ),
    ]
)


async def run_client(user_query: str):
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    client = Client(MCP_ENDPOINT)

    async with client:
        # --------------------------------------------------
        # 1. LIST TOOLS
        # --------------------------------------------------
        tools = await client.list_tools()
        tool_lines = []

        for t in tools:
            schema = getattr(t, "inputSchema", {}) or {}
            props = schema.get("properties", {}) or {}
            args = ", ".join(props.keys()) if props else "no_args"
            tool_lines.append(f"- {t.name}({args})")

        tools_text = "\n".join(tool_lines)

        # --------------------------------------------------
        # 2. PLAN (MULTI-STEP)
        # --------------------------------------------------
        chain = PLANNER_PROMPT | llm | StrOutputParser()
        raw_plan = await chain.ainvoke(
            {
                "tools": tools_text,
                "query": user_query,
            }
        )

        # Strip markdown code blocks if present
        cleaned_plan = raw_plan.strip()
        if cleaned_plan.startswith("```"):
            # Remove opening ```json or ```
            lines = cleaned_plan.split('\n')
            lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            cleaned_plan = '\n'.join(lines)

        try:
            plan: List[Dict[str, Any]] = json.loads(cleaned_plan)
        except Exception:
            print("❌ Invalid planner output")
            print(raw_plan)
            return

        print("\n🧠 Execution Plan:")
        print(json.dumps(plan, indent=2))

        # --------------------------------------------------
        # 3. EXECUTE STEPS SEQUENTIALLY
        # --------------------------------------------------
        # --------------------------------------------------
        # 3. EXECUTE STEPS SEQUENTIALLY (ROBUST)
        # --------------------------------------------------
        previous_result: Any = None

        # Build tool index for schemas
        tool_index = {t.name: t for t in tools}

        for idx, step in enumerate(plan, start=1):
            tool_name = step["tool"]
            raw_args = step.get("args", {})

            tool = tool_index.get(tool_name)
            if not tool:
                raise RuntimeError(f"Tool '{tool_name}' not found")

            schema = getattr(tool, "inputSchema", {}) or {}
            properties = list((schema.get("properties") or {}).keys())

            # ---------------------------------------------
            # NORMALIZE ARGS (list → dict if needed)
            # ---------------------------------------------
            if isinstance(raw_args, list):
                if len(raw_args) != len(properties):
                    raise ValueError(
                        f"Argument count mismatch for tool '{tool_name}'"
                    )
                args = dict(zip(properties, raw_args))
            else:
                args = raw_args.copy()

            # Replace PREVIOUS_RESULT placeholders
            for k, v in args.items():
                if v == "PREVIOUS_RESULT":
                    args[k] = previous_result

            # Inject API key only if required
            if "openai_api_key" in properties:
                args["openai_api_key"] = OPENAI_API_KEY

            print(f"\n⚙️ Step {idx}: {tool_name}")
            print(f"Args: {args}")

            result = await client.call_tool(tool_name, args)
            previous_result = getattr(result, "data", None)

            print(f"Result: {previous_result}")

        print("\n✅ Final Output:", previous_result)


if __name__ == "__main__":
    asyncio.run(
        run_client("first add 5 and 8 then multiply it by 6")
    )
