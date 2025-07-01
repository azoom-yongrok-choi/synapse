import logging
import traceback
import os

# from toolbox_core import ToolboxSyncClient
from google.adk.agents.base_agent import BaseAgent
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from pydantic import PrivateAttr
from .classifier_agent import ClassifierAgent
from .parking_agent import ParkingAgent
from .common_agent import CommonAgent
from .tone_polish_agent import TonePolishAgent
from .parsing_agent import ParsingAgent
from .utils import RequestType
from .auth_agent import AuthAgent
from .custom_adk_patches import CustomMCPToolset


class BackOfficeRootAgent(BaseAgent):
    _parking_agent: ParkingAgent = PrivateAttr()
    _common_agent: CommonAgent = PrivateAttr()
    _classifier_agent: ClassifierAgent = PrivateAttr()
    _tone_polish_agent: TonePolishAgent = PrivateAttr()
    _auth_agent: AuthAgent = PrivateAttr()
    _parsing_agent: ParsingAgent = PrivateAttr()

    def __init__(self, ctx):
        logging.info("[ROOT AGENT] Initializing")

        super().__init__(name="main_agent")
        username = os.getenv("ES_USERNAME")
        password = os.getenv("ES_PASSWORD")
        es_url = os.getenv("ES_URL")
        # MCP tool import
        parking_tool = CustomMCPToolset(
            connection_params=StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@elastic/mcp-server-elasticsearch@0.1.1",
                ],
                env={
                    "ES_URL": es_url,
                    "ES_USERNAME": username,
                    "ES_PASSWORD": password,
                },
                # timeout=120, # It is not working 1.2.0
                # tool_filter=["search"],
            ),
        )

        dummy_tools = CustomMCPToolset(
            connection_params=StreamableHTTPServerParams(
                url="http://localhost:8080/mcp"
            ),
            tool_filter=["get-covid-json-keys"],
        )
        logging.info(f"[Dummy Tools] {[dummy_tools]}")
        # sub-agents
        self._classifier_agent = ClassifierAgent(ctx)
        self._parking_agent = ParkingAgent(ctx, tools=[parking_tool])
        self._common_agent = CommonAgent(ctx)
        self._tone_polish_agent = TonePolishAgent(ctx)
        self._auth_agent = AuthAgent(ctx)
        self._parsing_agent = ParsingAgent(ctx, tools=[dummy_tools])

    async def _run_async_impl(self, ctx):
        logging.info("[ROOT AGENT] Start workflow")
        logging.info(
            f"STATE: api_auth_success={ctx.session.state.get('api_auth_success')}, auth_in_progress={ctx.session.state.get('auth_in_progress')}, classifier_result={ctx.session.state.get('classifier_result')}"
        )

        # If authentication is in progress, treat user input as the authentication password and go directly to AuthAgent
        if ctx.session.state.get("auth_in_progress"):
            logging.info(
                "[ROOT AGENT] Detected authentication in progress. Treating user input as authentication password."
            )
            user_input = None
            # Save user input as user_auth_password
            if getattr(ctx, "user_content", None) and ctx.user_content.parts:
                user_input = ctx.user_content.parts[0].text
            logging.info(f"[ROOT AGENT] User input: {user_input}")
            if user_input:
                ctx.session.state["user_auth_password"] = user_input
            else:
                logging.info("[ROOT AGENT] No user input found.")
            async for event in self._auth_agent.run_async(ctx):
                yield event
            return

        # Run ClassifierAgent
        logging.info("[ROOT AGENT] Running ClassifierAgent")
        async for event in self._classifier_agent.run_async(ctx):
            yield event
        classifier_result = ctx.session.state.get("classifier_result")
        logging.info(f"[ROOT AGENT] Classifier result: {classifier_result}")

        # 2. Branch according to classification result
        if classifier_result == RequestType.PARKING:
            # Run AuthAgent
            logging.info(
                f"[ROOT AGENT] user_auth_password: {ctx.session.state.get('user_auth_password')}"
            )
            async for event in self._auth_agent.run_async(ctx):
                yield event
            api_auth_success = ctx.session.state.get("api_auth_success")
            if api_auth_success is False or api_auth_success is None:
                logging.info("[ROOT AGENT] Authentication failed or not authenticated.")
                return

            async for event in self._parking_agent.run_async(ctx):
                yield event
            response_text = ctx.session.state.get("response_text")
        elif classifier_result == RequestType.PARSING:
            async for event in self._parsing_agent.run_async(ctx):
                yield event
            response_text = ctx.session.state.get("response_text")
        else:
            async for event in self._common_agent.run_async(ctx):
                yield event
            response_text = ctx.session.state.get("response_text")
        logging.info("[ROOT AGENT] Response")

        # 3. TonePolishAgent
        ctx.session.state["to_polish"] = response_text
        async for event in self._tone_polish_agent.run_async(ctx):
            yield event
        polished_text = ctx.session.state.get("polished_text")
        logging.info("[ROOT AGENT] Polished")

        # 4. final response
        ctx.session.state["final_response"] = polished_text

        logging.info("[ROOT AGENT] Workflow finished")


root_agent = BackOfficeRootAgent(None)
