README.MD
mcp server client based weather app 
curl -LsSf https://astral.sh/uv/install.sh | sh

====MCP Weathher Server ============================

# Create a new directory for our project
uv init weather
cd weather

# Create virtual environment and activate it
uv venv
source .venv/bin/activate

# Install dependencies
uv add "mcp[cli]" httpx

# Create our server file
touch weather.py


build the server [ weather/weather.py ]


uv run weather.py

====MCP Weathher Client ============================
# Create project directory
uv init mcp-client
cd mcp-client

# Create virtual environment
uv venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate

# Install required packages
uv add mcp  python-dotenv
[ Refer requirements.txt for any version conflict or use pip install openai==0.28.0 ]
# Remove boilerplate files
# On Windows:
del main.py
# On Unix or MacOS:
rm main.py

# Create our main file
touch client.py

export OPENAI_API_KEY=sk-proj-*

uv run weather-client-gpt.py ../weather/weather.py
[uv run client.py path/to/server.py # python server]

