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

load_dotenv()  # load OPENAI_API_KEY from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        resp = await self.session.list_tools()
        tools = [tool.name for tool in resp.tools]
        print("Connected to server with tools:", tools)

    async def process_query(self, query: str) -> str:
        # 1) get tool definitions from MCP server
        resp = await self.session.list_tools()
        functions = []
        for t in resp.tools:
            functions.append({
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema
            })

        # 2) initial user message
        messages = [{"role": "user", "content": query}]

        # 3) call OpenAI ChatCompletion with function definitions
        completion = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            functions=functions,
            function_call="auto",
            max_tokens=1000
        )
        msg = completion.choices[0].message

        # 4) if GPT wants to call a tool, execute and then resume
        if msg.get("function_call"):
            fn_name = msg.function_call.name
            fn_args = json.loads(msg.function_call.arguments or "{}")

            # call the MCP tool
            tool_resp = await self.session.call_tool(fn_name, fn_args)

            # add the function call and its result to the conversation
            messages.append(msg)  # the assistant’s function_call
            messages.append({
                "role": "function",
                "name": fn_name,
                "content": tool_resp.content
            })

            # 5) get GPT’s final answer after the tool result
            followup = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000
            )
            return followup.choices[0].message.content

        # 6) else just return GPT’s reply
        return msg.content or ""

    async def chat_loop(self):
        print("MCP GPT-Client started. Type ‘quit’ to exit.")
        while True:
            query = input("Query: ").strip()
            if query.lower() == "quit":
                break
            try:
                resp = await self.process_query(query)
                print(resp)
            except Exception as e:
                print("Error:", e)

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

