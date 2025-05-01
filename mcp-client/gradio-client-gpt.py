import asyncio
import sys
import threading
import time
import json
import os
from typing import Optional
from contextlib import AsyncExitStack
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

import gradio as gr
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import openai
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.getenv('DOTENV_PATH', '.env')
load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a global event loop
event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(event_loop)

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.server_connected = False
        self.tools = []

    async def connect_to_server(self, server_script_path: str) -> str:
        try:
            if self.server_connected:
                await self.cleanup()

            if not server_script_path.endswith(('.py', '.js')):
                return "Error: Server script must be .py or .js"

            cmd = "python" if server_script_path.endswith('.py') else "node"
            params = StdioServerParameters(command=cmd, args=[server_script_path], env=None)

            transport = await self.exit_stack.enter_async_context(stdio_client(params))
            stdio, write = transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await self.session.initialize()

            resp = await self.session.list_tools()
            self.tools = resp.tools
            self.server_connected = True
            info = [f"{t.name}: {t.description}" for t in self.tools]
            return "✅ Connected with tools:\n" + "\n".join(info)
        except Exception as e:
            return f"❌ Connection error: {e}"

    async def process_query(self, query: str) -> str:
        if not self.server_connected or not self.session:
            return "❌ Not connected."
        try:
            # Prepare function definitions
            resp = await self.session.list_tools()
            functions = [
                {"name": t.name, "description": t.description, "parameters": t.inputSchema}
                for t in resp.tools
            ]

            # Call OpenAI ChatCompletion
            chat_resp = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": query}],
                functions=functions,
                function_call="auto"
            )
            msg = chat_resp.choices[0].message

            # If function call requested
            if msg.function_call:
                fn_name = msg.function_call.name
                args = json.loads(msg.function_call.arguments or "{}")
                tool_res = await self.session.call_tool(fn_name, args)

                # Normalize tool output to string
                raw_content = tool_res.content
                if isinstance(raw_content, list):
                    # extract text from any TextContent-like items
                    pieces = []
                    for item in raw_content:
                        # item may have .text or .content
                        text = getattr(item, 'text', None) or getattr(item, 'content', None) or str(item)
                        pieces.append(text)
                    normalized = "\n".join(pieces)
                elif hasattr(raw_content, 'content'):
                    normalized = raw_content.content
                else:
                    normalized = str(raw_content)

                # second ChatCompletion with function result
                follow = await openai.ChatCompletion.acreate(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"user","content":query},
                        {"role":"assistant","function_call":msg.function_call.to_dict()},
                        {"role":"function","name":fn_name,"content": normalized}
                    ]
                )
                return follow.choices[0].message.content or ""

            return msg.content or ""
        except Exception as e:
            return f"❌ Error: {e}"

    async def cleanup(self):
        if self.session:
            await self.exit_stack.aclose()
            self.server_connected = False
            self.session = None

# Instantiate client
client = MCPClient()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, event_loop).result()

# Gradio callbacks
def connect(server_path):
    return run_async(client.connect_to_server(server_path))
def disconnect():
    run_async(client.cleanup())
    return "Disconnected"

def chat(query, history):
    history = history or []
    history.append({"role": "user", "content": query})
    resp = run_async(client.process_query(query))
    history.append({"role": "assistant", "content": resp})
    return history

# Start event loop thread
def start_loop():
    asyncio.set_event_loop(event_loop)
    event_loop.run_forever()
threading.Thread(target=start_loop, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("# MCP Weather Client (OpenAI)")
    with gr.Row():
        path = gr.Textbox(label="Server script path", value="../weather/weather.py")
        connect_btn = gr.Button("Connect")
        disconnect_btn = gr.Button("Disconnect")
    status = gr.Textbox(label="Status", interactive=False)
    chatbox = gr.Chatbot(type="messages")
    inp = gr.Textbox(label="Your question")
    send_btn = gr.Button("Send")

    connect_btn.click(connect, inputs=path, outputs=status)
    disconnect_btn.click(disconnect, outputs=status)
    inp.submit(chat, inputs=[inp, chatbox], outputs=chatbox)
    send_btn.click(chat, inputs=[inp, chatbox], outputs=chatbox).then(lambda: None, None, inp)

if __name__ == "__main__":
    demo.launch()

