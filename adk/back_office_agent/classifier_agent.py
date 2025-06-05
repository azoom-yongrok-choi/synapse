from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from back_office_agent.utils import RequestType
import logging


class ClassifierAgent(LlmAgent):
    def __init__(self, ctx):
        logging.info("[ClassifierAgent] Initializing ClassifierAgent")

        super().__init__(
            name="classifier_agent",
            model=LiteLlm(model="openai/gpt-4o-mini"),
            instruction=f"""
Guidelines:
- You are a request classifier. 
- If the user message is about finding a parking lot (such as asking for location, searching, or where to park), output exactly '{RequestType.PARKING.value}'. 
- For all other cases, output exactly '{RequestType.OTHER.value}'. 
- Do not explain. Output only one word: '{RequestType.PARKING.value}' or '{RequestType.OTHER.value}'.
""",
            output_key="classifier_result",
        )
