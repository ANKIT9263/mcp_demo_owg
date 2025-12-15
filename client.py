import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Callable

from dotenv import load_dotenv
from fastmcp.client import Client

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class MCPAgentOrchestrator:
    """
    Orchestrates multi-step tool execution via MCP server
    """

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
                    'use the string "PREVIOUS_RESULT" (with quotes) as a value\n'
                    "- Always quote string values in JSON\n"
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

    def __init__(
        self,
        mcp_endpoint: str = "http://localhost:8080/mcp",
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
    ):
        """Initialize the orchestrator"""
        self.mcp_endpoint = mcp_endpoint
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL")
        self.stream_callback: Optional[Callable] = None

    async def _emit(self, event_type: str, data: Dict[str, Any]):
        """Emit event to stream callback if provided"""
        if self.stream_callback:
            await self.stream_callback(event_type, data)

    def _format_tools_list(self, tools: List) -> str:
        """Format tools into human-readable list for LLM"""
        tool_lines = []
        for tool in tools:
            schema = getattr(tool, "inputSchema", {}) or {}
            props = schema.get("properties", {}) or {}
            args = ", ".join(props.keys()) if props else "no_args"
            tool_lines.append(f"- {tool.name}({args})")
        return "\n".join(tool_lines)

    def _parse_plan_response(self, raw_plan: str) -> List[Dict[str, Any]]:
        """Parse and clean LLM response, handling markdown code blocks"""
        cleaned_plan = raw_plan.strip()

        # Strip markdown code blocks if present
        if cleaned_plan.startswith("```"):
            lines = cleaned_plan.split('\n')
            lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            cleaned_plan = '\n'.join(lines)

        # Fix unquoted PREVIOUS_RESULT (convert to string)
        import re
        cleaned_plan = re.sub(
            r':\s*PREVIOUS_RESULT\s*([,\]\}])',
            r': "PREVIOUS_RESULT"\1',
            cleaned_plan
        )
        cleaned_plan = re.sub(
            r'\[\s*PREVIOUS_RESULT\s*([,\]])',
            r'["PREVIOUS_RESULT"\1',
            cleaned_plan
        )

        return json.loads(cleaned_plan)

    async def generate_plan(
        self,
        query: str,
        tools: List,
        tools_text: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate execution plan from user query"""
        llm = ChatOpenAI(
            model=self.openai_model,
            temperature=0,
            api_key=self.openai_api_key,
        )

        chain = self.PLANNER_PROMPT | llm | StrOutputParser()
        raw_plan = await chain.ainvoke({
            "tools": tools_text,
            "query": query,
        })

        try:
            plan = self._parse_plan_response(raw_plan)
            print("\n🧠 Execution Plan:")
            print(json.dumps(plan, indent=2))
            await self._emit("plan", {"plan": plan})
            return plan
        except Exception as e:
            error_msg = f"❌ Invalid planner output: {str(e)}"
            print(error_msg)
            print(raw_plan)
            await self._emit("error", {"message": error_msg, "raw_output": raw_plan})
            return None

    def _normalize_args(
        self,
        raw_args: Any,
        properties: List[str],
        tool_name: str
    ) -> Dict[str, Any]:
        """Normalize arguments from list to dict format"""
        if isinstance(raw_args, list):
            if len(raw_args) != len(properties):
                raise ValueError(
                    f"Argument count mismatch for tool '{tool_name}'"
                )
            return dict(zip(properties, raw_args))
        return raw_args.copy()

    def _inject_dependencies(
        self,
        args: Dict[str, Any],
        previous_result: Any,
        properties: List[str]
    ) -> Dict[str, Any]:
        """Inject previous results and required dependencies"""
        # Replace PREVIOUS_RESULT placeholders
        for k, v in args.items():
            if v == "PREVIOUS_RESULT":
                args[k] = previous_result

        # Inject API key if required
        if "openai_api_key" in properties:
            args["openai_api_key"] = self.openai_api_key

        return args

    async def execute_step(
        self,
        client: Client,
        step: Dict[str, Any],
        step_number: int,
        tool_index: Dict[str, Any],
        previous_result: Any
    ) -> Any:
        """Execute a single tool step"""
        tool_name = step["tool"]
        raw_args = step.get("args", {})

        tool = tool_index.get(tool_name)
        if not tool:
            raise RuntimeError(f"Tool '{tool_name}' not found")

        schema = getattr(tool, "inputSchema", {}) or {}
        properties = list((schema.get("properties") or {}).keys())

        # Normalize and inject dependencies
        args = self._normalize_args(raw_args, properties, tool_name)
        args = self._inject_dependencies(args, previous_result, properties)

        # Mask sensitive data for display
        display_args = args.copy()
        if "openai_api_key" in display_args:
            key = display_args["openai_api_key"]
            display_args["openai_api_key"] = f"{key[:3]}..." if key else "***"

        print(f"\n⚙️ Step {step_number}: {tool_name}")
        print(f"Args: {display_args}")
        await self._emit("step", {
            "step": step_number,
            "tool": tool_name,
            "args": display_args
        })

        # Execute tool
        result = await client.call_tool(tool_name, args)
        result_data = getattr(result, "data", None)

        print(f"Result: {result_data}")
        await self._emit("step_result", {
            "step": step_number,
            "result": result_data
        })

        return result_data

    async def execute_plan(
        self,
        client: Client,
        plan: List[Dict[str, Any]],
        tools: List
    ) -> Any:
        """Execute the full plan sequentially"""
        previous_result: Any = None
        tool_index = {t.name: t for t in tools}

        for idx, step in enumerate(plan, start=1):
            previous_result = await self.execute_step(
                client, step, idx, tool_index, previous_result
            )

        print("\n✅ Final Output:", previous_result)
        await self._emit("final", {"result": previous_result})

        return previous_result

    async def run(
        self,
        query: str,
        stream_callback: Optional[Callable] = None
    ) -> Any:
        """
        Main entry point: Run agent with query

        Args:
            query: User query to process
            stream_callback: Optional async callback(event_type: str, data: dict)

        Returns:
            Final execution result
        """
        self.stream_callback = stream_callback
        client = Client(self.mcp_endpoint)

        async with client:
            # List available tools
            tools = await client.list_tools()
            tools_text = self._format_tools_list(tools)

            # Generate execution plan
            plan = await self.generate_plan(query, tools, tools_text)
            if not plan:
                return None

            # Execute plan
            result = await self.execute_plan(client, plan, tools)
            return result


# Convenience function for backward compatibility
async def run_client(user_query: str, stream_callback=None):
    """
    Legacy function wrapper for backward compatibility
    """
    orchestrator = MCPAgentOrchestrator()
    return await orchestrator.run(user_query, stream_callback)


if __name__ == "__main__":
    async def main():
        orchestrator = MCPAgentOrchestrator()
        result = await orchestrator.run(
            "first add 5 and 8 then multiply it by 6"
        )
        print(f"\n🎯 Final Result: {result}")

    asyncio.run(main())
