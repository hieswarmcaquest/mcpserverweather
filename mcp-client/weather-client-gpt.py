import asyncio
import json
import threading
import queue
import os
import sys
from contextlib import AsyncExitStack
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox

import openai
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class MCPClient:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def connect(self, script_path: str):
        is_py = script_path.endswith('.py')
        is_js = script_path.endswith('.js')
        if not (is_py or is_js):
            raise ValueError("Server script must be .py or .js file")
        cmd = "python" if is_py else "node"
        params = StdioServerParameters(command=cmd, args=[script_path], env=None)
        transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

    async def process(self, query: str) -> str:
        resp = await self.session.list_tools()
        functions = [{"name": t.name, "description": t.description, "parameters": t.inputSchema}
                     for t in resp.tools]
        messages = [{"role": "user", "content": query}]
        completion = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            functions=functions,
            function_call="auto",
            max_tokens=1000
        )
        msg = completion.choices[0].message
        if msg.get("function_call"):
            fn = msg.function_call.name
            args = json.loads(msg.function_call.arguments or "{}")
            tool_resp = await self.session.call_tool(fn, args)
            messages.append({"role": "assistant", "content": None, "function_call": {"name": fn, "arguments": msg.function_call.arguments}})
            messages.append({"role": "function", "name": fn, "content": tool_resp.content})
            followup = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini", messages=messages, max_tokens=1000
            )
            return followup.choices[0].message.content or ""
        return msg.content or ""

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MCP Weather GPT Client")
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.chat = scrolledtext.ScrolledText(self.frame, state='disabled', wrap=tk.WORD)
        self.chat.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.input_frame = tk.Frame(self.frame)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)
        self.entry = tk.Entry(self.input_frame)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind('<Return>', lambda e: self.send_query())
        self.send_btn = tk.Button(self.input_frame, text="Send", command=self.send_query)
        self.send_btn.pack(side=tk.RIGHT)

        self.menu = tk.Menu(root)
        root.config(menu=self.menu)
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Connect to Server…", command=self.choose_and_connect)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)

        self.query_q = queue.Queue()
        self.response_q = queue.Queue()
        self.client = MCPClient()
        self.connected = False
        self.root.after(100, self.check_responses)

    def log(self, msg: str):
        self.chat.configure(state='normal')
        self.chat.insert(tk.END, msg + '
')
        self.chat.configure(state='disabled')
        self.chat.yview(tk.END)

    def choose_and_connect(self):
        path = filedialog.askopenfilename(title="Select server script (.py/.js)", filetypes=[("Python/JS files","*.py *.js")])
        if not path:
            return
        self.log(f"Connecting to server: {path} ...")
        threading.Thread(target=self.start_async_loop, args=(path,), daemon=True).start()

    def start_async_loop(self, script_path):
        asyncio.run(self.async_main(script_path))

    async def async_main(self, script_path):
        try:
            await self.client.connect(script_path)
            self.connected = True
            self.log("✅ Connected to MCP server.")
            while True:
                query = await asyncio.get_event_loop().run_in_executor(None, self.query_q.get)
                self.log(f"You: {query}")
                try:
                    resp = await self.client.process(query)
                    self.response_q.put(resp)
                except Exception as e:
                    self.response_q.put(f"Error: {e}")
        except Exception as e:
            self.log(f"Connection failed: {e}")

    def send_query(self):
        if not self.connected:
            messagebox.showwarning("Not connected", "Please connect to a server first.")
            return
        q = self.entry.get().strip()
        if not q:
            return
        self.entry.delete(0, tk.END)
        self.query_q.put(q)

    def check_responses(self):
        while not self.response_q.empty():
            resp = self.response_q.get()
            self.log(f"GPT: {resp}")
        self.root.after(100, self.check_responses)

if __name__ == '__main__':
    root = tk.Tk()
    GUI(root)
    root.mainloop()
