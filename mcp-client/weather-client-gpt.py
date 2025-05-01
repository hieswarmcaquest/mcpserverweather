import sys
import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import os
import openai
from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        is_py = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_py or is_js):
            raise ValueError("Server script must be .py or .js")

        cmd = "python" if is_py else "node"
        params = StdioServerParameters(command=cmd, args=[server_script_path], env=None)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        resp = await self.session.list_tools()
        print("Connected to server with tools:", [t.name for t in resp.tools])

    async def process_query(self, query: str) -> str:
        # Fetch tool definitions
        resp = await self.session.list_tools()
        functions = [
            {"name": t.name, "description": t.description, "parameters": t.inputSchema}
            for t in resp.tools
        ]

        # Initial user message
        messages = [{"role": "user", "content": query}]

        # Ask GPT
        completion = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            functions=functions,
            function_call="auto",
            max_tokens=1000
        )
        msg = completion.choices[0].message

        if msg.get("function_call"):
            # Execute the requested tool
            fn = msg.function_call.name
            args = json.loads(msg.function_call.arguments or "{}")
            tool_resp = await self.session.call_tool(fn, args)

            # Debugging Content Type
            print("Debug - Content Type:", type(tool_resp.content))
            print("Debug - Content Type index:", type(tool_resp.content[0]))
            print("Debug - Content:", tool_resp.content)

            # Extract text from TextContent objects within lists
            content = tool_resp.content
            if isinstance(content, list):
                content = [item.text if hasattr(item, "text") else str(item) for item in content]
            elif isinstance(content, dict):
                raw = json.dumps(content)
            else:
                raw = json.dumps({"response": str(content)})

            raw = json.dumps(content)  # Ensure final content is JSON serializable

            # Append the assistant’s function_call as a pure dict
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": fn,
                    "arguments": msg.function_call.arguments,
                }
            })

            # Append the function’s response
            messages.append({"role": "function", "name": fn, "content": raw})

            # Get GPT’s final answer
            followup = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000
            )
            return followup.choices[0].message.content or ""

        # No function call: just return the assistant’s reply
        return msg.content or ""

    async def chat_loop(self):
        print("MCP GPT-Client started. Type ‘quit’ to exit.")
        while True:
            q = input("Query: ").strip()
            if q.lower() == "quit":
                break
            try:
                print(await self.process_query(q))
            except Exception as e:
                print("Error:", e)

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python weather-client-gpt.py <path_to_server_script>")
        sys.exit(1)
    c = MCPClient()
    try:
        await c.connect_to_server(sys.argv[1])
        await c.chat_loop()
    finally:
        await c.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

