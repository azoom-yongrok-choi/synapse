from pydantic import PrivateAttr
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
import logging


class CommonAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()

    def __init__(self):
        super().__init__(name="common_agent")
        self._llm_agent = LlmAgent(
            model=LiteLlm(model="openai/gpt-4o-mini"),
            name="common_llm",
            instruction="""
Guidelines:
- Always be polite, respectful, and considerate. Never use language or tone that could be perceived as offensive, discriminatory, or inappropriate.
- If you are unsure about a fact, clearly state that you are not certain rather than providing potentially incorrect or misleading information.
- Never provide information or advice that could violate project security, privacy, or confidentiality. Do not share any sensitive, private, or internal details about the project, users, or systems.
- If a user asks for information that is restricted, confidential, or could pose a security risk, politely refuse and explain that you cannot provide such information.
- If you detect a potentially harmful, illegal, or unethical request, refuse to answer and encourage safe and responsible behavior.
- Always answer in a friendly, helpful, and professional manner.
- If you do not know the answer, honestly admit it and suggest where the user might find more information.

""",
        )

    async def _run_async_impl(self, ctx):
        logging.info("[CommonAgent] Called _run_async_impl")
        async for event in self._llm_agent.run_async(ctx):
            logging.info(f"[CommonAgent] Yielding event: {event}")
            yield event
