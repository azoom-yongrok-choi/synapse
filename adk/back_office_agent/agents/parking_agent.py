import logging
from pydantic import PrivateAttr
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from ..utils import (
    get_default_parking_fields,
    get_nested_fields,
    ensure_required_params_callback,
)

DEFAULT_PARKING_FIELDS = get_default_parking_fields()
NESTED_FIELDS = get_nested_fields()


class ParkingAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()

    def __init__(self, tools):
        super().__init__(name="parking_agent")
        fields_str = ", ".join(DEFAULT_PARKING_FIELDS)
        nested_fields_str = ", ".join(NESTED_FIELDS)
        self._llm_agent = LlmAgent(
            model=LiteLlm(model="openai/gpt-4o-mini"),
            name="parking_llm",
            instruction=f"""
You are a helpful AI agent specialized in parking lot search. For every user request, you MUST use the Elasticsearch MCP server's search tool to retrieve real data. 

Guidelines:
- For any data query, NEVER answer from your own knowledge or imagination. ALWAYS call the MCP search tool and return only the tool's result to the user.
- All data queries must use only the 'parking' index. Never use any other index.
- Every MCP search tool call must include a valid 'queryBody' parameter.
- When building 'queryBody':
  - Use these fields by default unless the user specifies otherwise: {fields_str}
  - For text fields, use match_phrase queries.
  - For keyword fields, use term queries.
  - For boolean fields, use term queries with true/false.
  - For long/float fields, use range or term queries.
  - For date fields, use range queries.
  - For nested fields ({nested_fields_str}), always use the Elasticsearch nested query structure.

Example nested query:
{{
  "queryBody": {{
    "query": {{
      "nested": {{
        "path": "nearbyStations",
        "query": {{
          "match_phrase": {{ "nearbyStations.name": "保谷" }}
        }}
      }}
    }}
  }}
}}
""",
            tools=tools,
            before_tool_callback=ensure_required_params_callback,
            # tool_call_mode="auto",  # Uncomment if your ADK/LiteLLM version supports this option
        )

    async def _run_async_impl(self, ctx):
        logging.info("[ParkingAgent] Called _run_async_impl")
        async for event in self._llm_agent.run_async(ctx):
            logging.info(f"[ParkingAgent] Yielding event: {event}")
            yield event
