import logging
import os
from enum import Enum
from pydantic import PrivateAttr
from .agents.parking_agent import ParkingAgent
from .agents.common_agent import CommonAgent
from .agents.tone_polish_agent import TonePolishAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters


class RequestType(Enum):
    PARKING = "parking"
    OTHER = "other"


class MainAgent(BaseAgent):
    _parking_agent: ParkingAgent = PrivateAttr()
    _common_agent: CommonAgent = PrivateAttr()
    _classifier_llm: LlmAgent = PrivateAttr()
    _tone_polish_agent: TonePolishAgent = PrivateAttr()

    def __init__(self, tools):
        super().__init__(name="main_agent")
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
            output_key="classifier_label",
        )
        self._parking_agent = ParkingAgent(tools)
        self._common_agent = CommonAgent()
        self._tone_polish_agent = TonePolishAgent()

    async def _run_async_impl(self, ctx):
        # 1. 분류
        async for _ in self._classifier_llm.run_async(ctx):
            pass
        label = ctx.session.state.get("classifier_label", "")
        try:
            request_type = RequestType(label)
        except ValueError:
            request_type = RequestType.OTHER  # fallback for unexpected LLM output
        logging.info(f"[Classifier] request_type: {request_type}")

        # 2. 분기
        if request_type == RequestType.PARKING:
            agent = self._parking_agent
        else:
            agent = self._common_agent

        # 3. 하위 에이전트 실행 및 답변 추출
        response_text = None
        async for event in agent.run_async(ctx):
            if (
                hasattr(event, "content")
                and event.content
                and hasattr(event.content, "parts")
                and event.content.parts
            ):
                response_text = event.content.parts[0].text
                break

        # 4. 다듬기
        ctx.session.state["to_polish"] = response_text
        async for polish_event in self._tone_polish_agent.run_async(ctx):
            yield polish_event


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
