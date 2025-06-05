import logging
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from back_office_agent.utils import (
    get_default_parking_fields,
    get_nested_fields,
    ensure_required_params_callback,
)

DEFAULT_PARKING_FIELDS = get_default_parking_fields()
NESTED_FIELDS = get_nested_fields()


class ParkingAgent(LlmAgent):
    def __init__(self, ctx, tools):
        logging.info("[ParkingAgent] Initializing ParkingAgent")

        fields_str = ", ".join(DEFAULT_PARKING_FIELDS)
        nested_fields_str = ", ".join(NESTED_FIELDS)

        super().__init__(
            name="parking_agent",
            model=LiteLlm(model="openai/gpt-4o-mini"),
            instruction=f"""
You are a helpful AI agent specialized in parking lot search. For every user request, you MUST use the Elasticsearch MCP server's search tool to retrieve real data. 

Guidelines:
- For every response, always start with a title line: `[Parking Agent]` (include this exactly, at the very top of your reply).
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
            output_key="response_text",
            tools=tools,
            # before_tool_callback=ensure_required_params_callback,
            # tool_call_mode="auto",  # Uncomment if your ADK/LiteLLM version supports this option
        )

    async def run_async(self, ctx):
        # 1. 프롬프트/명령어 로그
        logging.info("###### PARKING AGENT ######")
        logging.info(f"[ParkingAgent] LLM instruction: {self.instruction}")
        logging.info(f"[ParkingAgent] Tools: {self.tools}")

        async for event in super().run_async(ctx):
            # 2. tool call 결과 로그
            if hasattr(event.actions, "tool_calls"):
                logging.info(f"[ParkingAgent] Tool calls: {event.actions.tool_calls}")
            # 3. state_delta(최종 응답) 로그
            logging.info(
                f"[ParkingAgent] state_delta: {getattr(event.actions, 'state_delta', None)}"
            )
            # 4. 전체 event 로그
            logging.info(f"[ParkingAgent] Full event: {event}")
            yield event
