import logging
import os
from enum import Enum
from pydantic import PrivateAttr
from .parking_agent import ParkingAgent
from .common_agent import CommonAgent
from .utils import (
    get_default_parking_fields,
    get_nested_fields,
)
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters


# Dynamically extract fields from data_type.json
DEFAULT_PARKING_FIELDS = get_default_parking_fields()
NESTED_FIELDS = get_nested_fields()

AGENT_PROMPT_COMMON_RULES = """
- Always respond in the user's language. Detect the user's language from their last message and answer in that language.
- If the user's message contains any Korean characters, always answer in Korean, even if there are Japanese or other language words mixed in.
- Use a friendly and approachable tone, as if you are chatting with a friend.
- Use emojis in your answers to make the conversation more fun and friendly! 🚗🅿️✨
"""


class UserFriendlyToolError(Exception):
    pass


class RequestType(Enum):
    PARKING = "parking"
    OTHER = "other"


class MainAgent(BaseAgent):
    _parking_agent: ParkingAgent = PrivateAttr()
    _common_agent: CommonAgent = PrivateAttr()
    _classifier_llm: LlmAgent = PrivateAttr()

    def __init__(self, tools):
        super().__init__(name="main_agent")
        self._parking_agent = ParkingAgent(tools)
        self._common_agent = CommonAgent()
        self._classifier_llm = LlmAgent(
            model=LiteLlm(model="openai/gpt-4o-mini"),
            name="classifier_llm",
            instruction=(
                "You are a request classifier. "
                "The user message to classify is stored in session state with key 'classifier_prompt'. "
                f"If the user message is about finding a parking lot (such as asking for location, searching, or where to park), output exactly '{RequestType.PARKING.value}'. "
                f"For all other cases, output exactly '{RequestType.OTHER.value}'. "
                f"Do not explain. Output only one word: '{RequestType.PARKING.value}' or '{RequestType.OTHER.value}'."
            ),
        )

    async def _run_async_impl(self, ctx):
        user_input = ""
        if hasattr(ctx, "session") and hasattr(ctx.session, "state"):
            user_input = ctx.session.state.get("last_user_message", "")
        ctx.session.state["classifier_prompt"] = user_input  # just the message
        label = ""
        async for event in self._classifier_llm.run_async(ctx):
            # Try to extract text from event.content.parts[0].text
            if hasattr(event, "content") and event.content:
                parts = getattr(event.content, "parts", None)
                if parts and len(parts) > 0 and hasattr(parts[0], "text"):
                    label = parts[0].text.strip().lower()
                elif hasattr(event.content, "text") and callable(event.content.text):
                    label = event.content.text().strip().lower()
                elif hasattr(event.content, "text"):
                    label = event.content.text.strip().lower()
                else:
                    label = str(event.content).strip().lower()
            else:
                label = ""
            break  # Only use the first event
        logging.info(f"[Classifier] label: {label}")
        try:
            request_type = RequestType(label)
        except ValueError:
            request_type = RequestType.OTHER  # fallback for unexpected LLM output
        logging.info(f"[Classifier] request_type: {request_type}")
        if request_type == RequestType.PARKING:
            logging.info("Routing to parking_agent")
            async for event in self._parking_agent.run_async(ctx):
                yield event
        else:
            logging.info("Routing to common_agent")
            async for event in self._common_agent.run_async(ctx):
                yield event


# Create MCPToolset from environment variables
mcp_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="npx",
        args=["-y", "@elastic/mcp-server-elasticsearch"],
        env={
            "ES_URL": os.getenv("ES_URL"),
            "ES_USERNAME": os.getenv("ES_USERNAME"),
            "ES_PASSWORD": os.getenv("ES_PASSWORD"),
        },
    )
)

root_agent = MainAgent([mcp_toolset])
