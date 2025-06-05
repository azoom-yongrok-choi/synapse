# Demo for AI Agent
It's a simple demo for AI agent(PJT-Synapse)

### Core stacks
- ADK : for AI pipeline
- Open AI : for LLM agent
- MCP : for connect data
- Elasticsearch : for data query

## How to start

### Set up
```bash
# make virtual env
python -m venv venv 
# if cannot work upper code(ex. mac os)
python3 -m venv venv 

# activate virtual env
source venv/bin/activate 
# install requirement
pip install -r requirements.txt 
```

### `.env`
You must need to set environment variables file
Plz ask to "Choi Yongrok" for sharing dashline

_It must be located in `/back_office_agent`_

### Run code
```bash
adk web
```

### Run MCP server
```bash
./toolbox --tools-file "tools.yaml"
```