import logging
import os
from toolbox_core import ToolboxSyncClient
from google.adk.agents.base_agent import BaseAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from pydantic import PrivateAttr
from .classifier_agent import ClassifierAgent
from .parking_agent import ParkingAgent
from .common_agent import CommonAgent
from .tone_polish_agent import TonePolishAgent
from .utils import RequestType


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
        # MCP tool import
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
                timeout=120,
            )
        )

        toolbox_url = os.environ.get("TOOLBOX_URL", "http://127.0.0.1:5000")
        # toolbox_url = os.environ.get("TOOLBOX_URL", "http://mcp:5001")
        toolbox = ToolboxSyncClient(toolbox_url)
        dummy_tools = toolbox.load_toolset("dummy-toolset")

        # sub-agents
        self._classifier_agent = ClassifierAgent(ctx)
        self._parking_agent = ParkingAgent(ctx, tools=[parking_tool])
        self._common_agent = CommonAgent(ctx, tools=dummy_tools)
        self._tone_polish_agent = TonePolishAgent(ctx)

    async def _run_async_impl(self, ctx):
        logging.info("[BackOfficeRootAgent] Start workflow")
        logging.info(
            f"STATE: api_auth_success={ctx.session.state.get('api_auth_success')}, auth_in_progress={ctx.session.state.get('auth_in_progress')}, classifier_result={ctx.session.state.get('classifier_result')}"
        )

        # ClassifierAgent 실행
        logging.info("[BackOfficeRootAgent] ClassifierAgent 실행")
        async for event in self._classifier_agent.run_async(ctx):
            yield event
        classifier_result = ctx.session.state.get("classifier_result")
        logging.info(f"[BackOfficeRootAgent] Classifier result: {classifier_result}")

        # 2. 분류 결과에 따라 분기
        if classifier_result == RequestType.PARKING:
            logging.info("[BackOfficeRootAgent] classifier_result=parking")
            async for event in self._parking_agent.run_async(ctx):
                yield event
            response_text = ctx.session.state.get("response_text")
        else:
            logging.info(
                f"[BackOfficeRootAgent] classifier_result={classifier_result} → CommonAgent 실행"
            )
            async for event in self._common_agent.run_async(ctx):
                yield event
            response_text = ctx.session.state.get("response_text")
        logging.info(f"[BackOfficeRootAgent] Response text: {response_text}")

        # 3. TonePolishAgent
        ctx.session.state["to_polish"] = response_text
        async for event in self._tone_polish_agent.run_async(ctx):
            yield event
        polished_text = ctx.session.state.get("polished_text")
        logging.info(f"[BackOfficeRootAgent] Polished text: {polished_text}")

        # 4. final response
        ctx.session.state["final_response"] = polished_text

        logging.info("[BackOfficeRootAgent] Workflow finished")


root_agent = BackOfficeRootAgent(None)
