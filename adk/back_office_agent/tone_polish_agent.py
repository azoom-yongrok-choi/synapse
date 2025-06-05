from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
import logging


class TonePolishAgent(LlmAgent):
    def __init__(self, ctx):
        logging.info("[TonePolishAgent] Initializing TonePolishAgent")

        super().__init__(
            name="tone_polish_agent",
            model=LiteLlm(model="openai/gpt-4o-mini"),
            instruction="""
Guidelines:
- For every response, always start with a title line: `[Tone Polish Agent]` (include this exactly, at the very top of your reply).
- Please rewrite the user's message to sound more natural, friendly, and approachable.
- Use clear and concise language, and break up long sentences for easier reading.
- Add appropriate line breaks and spacing so the response is easy to scan and not tiring to read.
- If the message sounds too formal or unfriendly, make it warmer and more inviting.
- Preserve the user's original level of politeness (formal/informal speech).
- Remove unnecessary repetition or overly long sentences, making the message concise and clear.
- Always reply in the user's language.
- Use emojis and a friendly tone where appropriate to make the message more engaging! 😊🚗🅿️✨
""",
            output_key="polished_text",
        )
