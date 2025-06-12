import os
import logging
from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event, EventActions
from google.genai.types import Part, Content


class AuthAgent(BaseAgent):
    def __init__(self, ctx):
        super().__init__(name="auth_agent")
        self._expected_key = os.getenv("DEMO_AUTH_API_KEY")

    async def run_async(self, ctx):
        # If already authenticated, exit immediately
        if ctx.session.state.get("api_auth_success"):
            return

        # Check if user has entered a password
        user_password = ctx.session.state.get("user_auth_password")
        logging.info(f"[AuthAgent] User input password: {user_password}")
        if user_password is None:
            # Request password input
            ctx.session.state["auth_in_progress"] = True
            yield Event(
                author=self.name,
                content=Content(
                    parts=[Part(text="Please enter your password for authentication.")]
                ),
                actions=EventActions(state_delta={"auth_in_progress": True}),
            )
            return

        # Password verification
        if user_password == self._expected_key:
            ctx.session.state["api_auth_success"] = True
            ctx.session.state["auth_in_progress"] = False
            logging.info("[AuthAgent] Authentication successful!")
            yield Event(
                author=self.name,
                content=Content(
                    parts=[
                        Part(
                            text="Authentication successful. Please re-enter your request you want."
                        )
                    ]
                ),
                actions=EventActions(
                    state_delta={"api_auth_success": True, "auth_in_progress": False}
                ),
            )
        else:
            ctx.session.state["api_auth_success"] = False
            ctx.session.state["auth_in_progress"] = False
            logging.info("[AuthAgent] Authentication failed!")
            yield Event(
                author=self.name,
                content=Content(parts=[Part(text="Authentication failed.")]),
                actions=EventActions(
                    state_delta={"api_auth_success": False, "auth_in_progress": False}
                ),
            )
