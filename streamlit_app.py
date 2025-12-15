"""
Streamlit Chat Interface for MCP Agent
--------------------------------------
Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json
from typing import Generator

# Page config
st.set_page_config(
    page_title="MCP Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000/run_agent"

# Custom CSS for ChatGPT-like interface
st.markdown("""
<style>
    /* Full width layout */
    .main .block-container {
        max-width: 100%;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /* Chat messages styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #f7f7f8;
    }
    .assistant-message {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
    }
    .message-content {
        margin-top: 0.5rem;
    }

    /* Step containers */
    .step-container {
        background-color: #f0f9ff;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
        border-left: 3px solid #3b82f6;
    }
    .result-container {
        background-color: #f0fdf4;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
        border-left: 3px solid #22c55e;
    }
    .error-container {
        background-color: #fef2f2;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ef4444;
    }

    /* Chat input full width */
    .stChatInput {
        max-width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🤖 MCP Agent Chat")
st.caption("Multi-step tool orchestration powered by MCP")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Make streaming request to API
            response = requests.post(
                API_URL,
                json={"query": prompt},
                stream=True,
                headers={"Accept": "text/event-stream"}
            )

            if response.status_code == 200:
                current_plan = None
                steps_output = []
                final_result = None

                # Process SSE stream
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')

                        # Parse SSE format
                        if line.startswith('event:'):
                            event_type = line.split('event:')[1].strip()
                        elif line.startswith('data:'):
                            data = json.loads(line.split('data:')[1].strip())

                            # Handle different event types
                            if event_type == 'plan':
                                current_plan = data.get('plan', [])
                                plan_text = "**📋 Execution Plan:**\n"
                                for i, step in enumerate(current_plan, 1):
                                    plan_text += f"{i}. `{step['tool']}` with args: `{step['args']}`\n"
                                full_response += plan_text + "\n"
                                message_placeholder.markdown(full_response)

                            elif event_type == 'step':
                                step_num = data.get('step')
                                tool = data.get('tool')
                                args = data.get('args')
                                step_text = f'<div class="step-container">⚙️ <b>Step {step_num}:</b> Executing <code>{tool}</code> with {args}</div>'
                                full_response += step_text
                                message_placeholder.markdown(full_response, unsafe_allow_html=True)

                            elif event_type == 'step_result':
                                result = data.get('result')
                                result_text = f'<div class="result-container">✅ <b>Result:</b> {result}</div>'
                                full_response += result_text
                                message_placeholder.markdown(full_response, unsafe_allow_html=True)

                            elif event_type == 'final':
                                final_result = data.get('result')
                                final_text = f"\n\n**🎯 Final Answer:** `{final_result}`"
                                full_response += final_text
                                message_placeholder.markdown(full_response, unsafe_allow_html=True)

                            elif event_type == 'error':
                                error_msg = data.get('message', 'Unknown error')
                                error_text = f'<div class="error-container">❌ <b>Error:</b> {error_msg}</div>'
                                full_response += error_text
                                message_placeholder.markdown(full_response, unsafe_allow_html=True)

                            elif event_type == 'done':
                                break

            else:
                error_text = f'<div class="error-container">❌ <b>API Error:</b> Status {response.status_code}</div>'
                full_response = error_text
                message_placeholder.markdown(full_response, unsafe_allow_html=True)

        except Exception as e:
            error_text = f'<div class="error-container">❌ <b>Error:</b> {str(e)}</div>'
            full_response = error_text
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chat interface demonstrates:
    - Multi-step tool planning
    - Sequential tool execution
    - Real-time streaming updates

    **Powered by:**
    - FastMCP Server
    - LangChain
    - OpenAI GPT
    """)

    st.divider()

    st.header("🔧 Settings")
    st.text(f"API: {API_URL}")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.header("💡 Example Queries")
    examples = [
        "Add 5 and 8, then multiply by 6",
        "Calculate (10 + 5) * 3",
        "What is 100 divided by 4?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.temp_query = ex
            st.rerun()

# Handle example query click
if "temp_query" in st.session_state:
    query = st.session_state.temp_query
    del st.session_state.temp_query
    st.rerun()

if __name__ == '__main__':
    pass