# synapse
It's a simple demo for AI agent(PJT-Synapse)

### Core stacks
- ADK : for AI pipeline
- Open AI : for LLM agent
- toolbox : MCP for connect bigquery data(dummy)
- Elasticsearch : for data query(dummy)

## ADK

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

### Environment Varialbles 
You must need to set environment variables.

### Run code
```bash
adk web
```

## Toolbox(MCP server)

### Set up
[genai-toolbox](https://github.com/googleapis/genai-toolbox?tab=readme-ov-file#installing-the-server)
It has several ways to install.
Plz choose proper version what you want.

Below code example is for this case.
- Install the library by binary image
- Your environment is Mac

```bash
# download binary image
curl -O https://storage.googleapis.com/genai-toolbox/v0.6.0/darwin/arm64/toolbox
# change access permission
chmod +x toolbox
```

### Make `YAML` for MCP tools
[MCP tools](https://googleapis.github.io/genai-toolbox/getting-started/mcp_quickstart/)
``` yaml
sources:
  my-bigquery-source:
    kind: bigquery
    project: [PROJECT_ID]
    location: asia-northeast1
tools:
  search-all-hotels-dummy:
    kind: bigquery-sql
    source: my-bigquery-source
    description: Search for all hotels.
    statement: SELECT * FROM `dummy.hotels`;
toolsets:
  dummy-toolset:
    - search-all-hotels-dummy
```

### Run MCP server
```bash
./toolbox --tools-file "tools.yaml"
```