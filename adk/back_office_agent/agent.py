import logging
import os
from toolbox_core import ToolboxSyncClient
from google.adk.agents.base_agent import BaseAgent
from back_office_agent.classifier_agent import ClassifierAgent
from back_office_agent.parking_agent import ParkingAgent
from back_office_agent.common_agent import CommonAgent
from back_office_agent.tone_polish_agent import TonePolishAgent
from back_office_agent.utils import RequestType
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from pydantic import PrivateAttr


class BackOfficeRootAgent(BaseAgent):
    _parking_agent: ParkingAgent = PrivateAttr()
    _common_agent: CommonAgent = PrivateAttr()
    _classifier_agent: ClassifierAgent = PrivateAttr()
    _tone_polish_agent: TonePolishAgent = PrivateAttr()

    def __init__(self, ctx):
        logging.info("[BackOfficeRootAgent] Initializing root agent and sub-agents")

        super().__init__(name="main_agent")
        username = os.getenv("ES_USERNAME")
        password = os.getenv("ES_PASSWORD")
        es_url = os.getenv("ES_URL")
        # MCP tool 생성 (실제 MCP tool import 및 설정 필요)
        parking_tool = MCPToolset(
            connection_params=StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@elastic/mcp-server-elasticsearch",
                ],
                env={
                    "ES_URL": es_url,
                    "ES_USERNAME": username,
                    "ES_PASSWORD": password,
                },
                timeout=60,
            )
            # ... 필요한 설정 추가 ...
        )

        toolbox_url = os.environ.get("TOOLBOX_URL", "http://127.0.0.1:5000")
        toolbox = ToolboxSyncClient(toolbox_url)
        dummy_tools = toolbox.load_toolset("dummy-toolset")

        # 각 서브에이전트 인스턴스화
        self._classifier_agent = ClassifierAgent(ctx)
        self._parking_agent = ParkingAgent(ctx, tools=[parking_tool])
        self._common_agent = CommonAgent(ctx, tools=dummy_tools)
        self._tone_polish_agent = TonePolishAgent(ctx)

    async def _run_async_impl(self, ctx):
        logging.info("[BackOfficeRootAgent] Start workflow")

        # 1. ClassifierAgent 실행
        async for event in self._classifier_agent.run_async(ctx):
            yield event
        classifier_result = ctx.session.state.get("classifier_result")
        logging.info(f"[BackOfficeRootAgent] Classifier result: {classifier_result}")

        # 2. 분기
        if classifier_result == RequestType.PARKING:
            async for event in self._parking_agent.run_async(ctx):
                yield event
            response_text = ctx.session.state.get("response_text")
        else:
            async for event in self._common_agent.run_async(ctx):
                yield event
            response_text = ctx.session.state.get("response_text")
        logging.info(f"[BackOfficeRootAgent] Response text: {response_text}")

        # 3. TonePolishAgent 실행
        ctx.session.state["to_polish"] = response_text
        async for event in self._tone_polish_agent.run_async(ctx):
            yield event
        polished_text = ctx.session.state.get("polished_text")
        logging.info(f"[BackOfficeRootAgent] Polished text: {polished_text}")

        # 4. 최종 응답 저장
        ctx.session.state["final_response"] = polished_text

        logging.info("[BackOfficeRootAgent] Workflow finished")


root_agent = BackOfficeRootAgent(None)
# def root_agent(ctx):
#     return BackOfficeRootAgent(ctx)
