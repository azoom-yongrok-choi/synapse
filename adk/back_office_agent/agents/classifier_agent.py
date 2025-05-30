from pydantic import PrivateAttr
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from ..utils import RequestType
import logging


class ClassifierAgent(BaseAgent):
    _llm_agent: LlmAgent = PrivateAttr()

    def __init__(self):
        super().__init__(name="classifier_agent")
        self._llm_agent = LlmAgent(
            model=LiteLlm(model="openai/gpt-4o-mini"),
            name="classifier_llm",
            instruction=f"""
Guidelines:
- You are a request classifier. 
- If the user message is about finding a parking lot (such as asking for location, searching, or where to park), output exactly '{RequestType.PARKING.value}'. 
- For all other cases, output exactly '{RequestType.OTHER.value}'. 
- Do not explain. Output only one word: '{RequestType.PARKING.value}' or '{RequestType.OTHER.value}'.
""",
            output_key="classifier_label",
        )

    async def _run_async_impl(self, ctx):
        logging.info("[ClassifierAgent] Called _run_async_impl")
        async for event in self._llm_agent.run_async(ctx):
            logging.info(f"[ClassifierAgent] ctx.session.state: {ctx.session.state}")
            logging.info(f"[ClassifierAgent] Yielding event: {event}")
            yield event
