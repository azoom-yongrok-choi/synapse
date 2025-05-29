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
[LLM Response]

Guidelines:
- For every response, always start with a title line: `[LLM Response]` (include this exactly, at the very top of your reply).
- You are a helpful, friendly assistant. Always answer in the user's language and use a friendly, emoji-rich tone.
""",
        )

    async def _run_async_impl(self, ctx):
        logging.info("[CommonAgent] Called _run_async_impl")
        async for event in self._llm_agent.run_async(ctx):
            logging.info(f"[CommonAgent] Yielding event: {event}")
            yield event
