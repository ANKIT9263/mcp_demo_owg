# MCP Agent Demo

A multi-step tool orchestration system using the Model Context Protocol (MCP), LangChain, and OpenAI. The system automatically plans and executes multi-step workflows using available tools.

## 🌟 Features

- **Automatic Planning**: LLM-powered multi-step execution planning
- **Tool Orchestration**: Sequential tool execution with dependency management
- **Streaming API**: Real-time Server-Sent Events (SSE) streaming
- **Chat Interface**: ChatGPT-like UI built with Streamlit
- **Extensible**: Easy to add new tools

## 📋 Prerequisites

- Python 3.8+
- OpenAI API Key

## 🚀 Installation

1. **Clone the repository**
   ```bash
   cd /Users/ankit/Desktop/varsha_projects/mcp_demo_owg
   ```

2. **Install dependencies**
   ```bash
   pip install fastmcp langchain-openai python-dotenv uvicorn fastapi streamlit requests
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=sk-your-api-key-here
   OPENAI_MODEL=gpt-4o-mini
   ```

## 🎯 Quick Start

### Option 1: Using Streamlit UI (Recommended)

1. **Start the MCP Server** (Terminal 1)
   ```bash
   python server.py
   ```
   Server runs at: `http://localhost:8080/mcp`

2. **Start the API Server** (Terminal 2)
   ```bash
   python api.py
   ```
   API runs at: `http://localhost:8000`

3. **Start the Streamlit App** (Terminal 3)
   ```bash
   streamlit run streamlit_app.py
   ```
   UI opens at: `http://localhost:8501`

### Option 2: Using Python Client Directly

```bash
python client.py
```

### Option 3: Using API with curl

```bash
curl -X POST http://localhost:8000/run_agent \
  -H "Content-Type: application/json" \
  -d '{"query": "add 5 and 8 then multiply by 6"}'
```

## 📁 Project Structure

```
mcp_demo_owg/
├── server.py              # MCP server (loads and registers tools)
├── api.py                 # FastAPI server with SSE streaming
├── client.py              # MCPAgentOrchestrator class
├── streamlit_app.py       # Streamlit chat interface
├── tools/
│   └── generic_tools.py   # Math and conversational tools
├── .env                   # Environment variables (create this)
└── README.md             # This file
```

## 🛠️ Adding New Tools

### Step 1: Create a Tools File

Create a new file in the `tools/` directory (e.g., `tools/my_custom_tools.py`):

```python
"""
My Custom Tools for MCP
-----------------------
"""

from fastmcp import FastMCP

def register_tools(mcp: FastMCP):
    """Register custom tools with the MCP server"""

    @mcp.tool()
    async def greet_user(name: str) -> str:
        """Greet a user by name"""
        return f"Hello, {name}! Welcome to MCP Agent."

    @mcp.tool()
    async def calculate_square(number: float) -> float:
        """Calculate the square of a number"""
        return number ** 2

    @mcp.tool()
    async def reverse_text(text: str) -> str:
        """Reverse a given text"""
        return text[::-1]

    print("✅ Custom tools registered")
```

### Step 2: Register the Tools Module

Update `server.py` to load your new tools:

```python
def create_mcp_server() -> FastMCP:
    """Create a single MCP server and load all tool modules."""
    mcp = FastMCP("IntegratedTools")

    # Add your new module to this list
    tool_modules = [
        "generic_tools",
        "my_custom_tools"  # Add this line
    ]

    for module_name in tool_modules:
        # ... rest of the code
```

### Step 3: Restart the Server

Restart the MCP server to load the new tools:
```bash
python server.py
```

### Tool Function Requirements

- Must be `async` functions
- Use type hints for parameters
- Include docstrings (used by the planner)
- Decorate with `@mcp.tool()`
- Return serializable data (str, int, float, dict, list)

### Example: Tool with API Integration

```python
@mcp.tool()
async def fetch_weather(city: str, api_key: str) -> dict:
    """Fetch current weather for a city"""
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.weather.com/v1/current",
            params={"city": city, "key": api_key}
        )
        return response.json()
```

## 🧪 Testing

### Test with curl (SSE Stream)

```bash
curl -N -X POST http://localhost:8000/run_agent \
  -H "Content-Type: application/json" \
  -d '{
    "query": "first add 5 and 8 then multiply by 6"
  }'
```

**Expected Output:**
```
event: plan
data: {"plan": [{"tool": "add", "args": [5, 8]}, {"tool": "multiply", "args": ["PREVIOUS_RESULT", 6]}]}

event: step
data: {"step": 1, "tool": "add", "args": {"a": 5, "b": 8}}

event: step_result
data: {"step": 1, "result": 13}

event: step
data: {"step": 2, "tool": "multiply", "args": {"a": 13, "b": 6}}

event: step_result
data: {"step": 2, "result": 78}

event: final
data: {"result": 78}

event: done
data: {}
```

### Test with Postman

1. **Create a new POST request**
   - URL: `http://localhost:8000/run_agent`
   - Headers: `Content-Type: application/json`

2. **Request Body:**
   ```json
   {
     "query": "calculate (10 + 5) * 3"
   }
   ```

3. **View streaming response** in the Postman console

### Example Queries

```bash
# Math operations
curl -X POST http://localhost:8000/run_agent \
  -H "Content-Type: application/json" \
  -d '{"query": "what is 100 divided by 4?"}'

# Multi-step calculation
curl -X POST http://localhost:8000/run_agent \
  -H "Content-Type: application/json" \
  -d '{"query": "subtract 10 from 50, then multiply the result by 2"}'

# Using conversational tools
curl -X POST http://localhost:8000/run_agent \
  -H "Content-Type: application/json" \
  -d '{"query": "say hello to me"}'
```

## 🔧 API Reference

### POST /run_agent

Execute a multi-step agent workflow.

**Request:**
```json
{
  "query": "your natural language query here"
}
```

**Response:** Server-Sent Events (SSE) stream

**Event Types:**
- `plan` - Execution plan generated
- `step` - Tool execution started
- `step_result` - Tool execution completed
- `final` - Final result
- `error` - Error occurred
- `done` - Stream complete

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │
│  (Port 8501)    │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────┐
│   FastAPI       │
│   (Port 8000)   │ ◄─── SSE Stream
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MCPAgent        │
│ Orchestrator    │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   MCP Server    │
│   (Port 8080)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tool Modules   │
│  (generic, etc) │
└─────────────────┘
```

## 🔒 Security

- API keys are automatically masked in UI (displays as `sk-...`)
- Sensitive parameters are filtered from streaming responses
- Environment variables for credential management

## 🐛 Troubleshooting

### Issue: "Invalid planner output"
**Solution:** The LLM is not generating valid JSON. Check your OpenAI API key and model name in `.env`.

### Issue: "Connection refused" on port 8080
**Solution:** Make sure the MCP server is running (`python server.py`).

### Issue: "Module not found" error
**Solution:** Install missing dependencies:
```bash
pip install fastmcp langchain-openai python-dotenv uvicorn fastapi streamlit
```

## 📝 Available Tools (Default)

### Math Tools
- `add(a, b)` - Add two numbers
- `subtract(a, b)` - Subtract b from a
- `multiply(a, b)` - Multiply two numbers
- `divide(a, b)` - Divide a by b

### Conversational Tools
- `handle_greeting(text, openai_api_key)` - Respond to greetings

## 🤝 Contributing

To add new tool categories:
1. Create a new file in `tools/` directory
2. Implement `register_tools(mcp)` function
3. Add module name to `tool_modules` list in `server.py`
4. Restart the MCP server

## 📄 License

MIT License

## 🙏 Acknowledgments

- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [LangChain](https://langchain.com) - LLM orchestration
- [Streamlit](https://streamlit.io) - Web UI framework
