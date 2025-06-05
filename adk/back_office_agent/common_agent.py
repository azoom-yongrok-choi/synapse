from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
import logging


class CommonAgent(LlmAgent):
    def __init__(self, ctx, tools):
        logging.info("[CommonAgent] Initializing CommonAgent")
        logging.info(f"Tools: {tools}")

        super().__init__(
            name="common_agent",
            model=LiteLlm(model="openai/gpt-4o-mini"),
            instruction="""
Guidelines:
- For every response, always start with a title line: `[Common Agent]` (include this exactly, at the very top of your reply).
- If you are unsure about a fact, clearly state that you are not certain rather than providing potentially incorrect or misleading information.
- Never provide information or advice that could violate project security, privacy, or confidentiality. Do not share any sensitive, private, or internal details about the project, users, or systems.
- If a user asks for information that is restricted, confidential, or could pose a security risk, politely refuse and explain that you cannot provide such information.
- If you detect a potentially harmful, illegal, or unethical request, refuse to answer and encourage safe and responsible behavior.
- If you do not know the answer, honestly admit it and suggest where the user might find more information.

- If a user asks about Hotel Searches:
    - If you can answer directly, do so.
    - If you need more information or need to search, use the provided tools (search-all-hotels-dummy, search-hotels-by-name, search-hotels-by-location).
    - If tool usage is not possible, answer as best as you can based on your knowledge.
""",
            output_key="response_text",
            tools=tools,
        )

    async def run_async(self, ctx):
        async for event in super().run_async(ctx):
            logging.info(f"[CommonAgent] Done: {event.actions}")
            logging.info(f"[CommonAgent] event(repr): {repr(event)}")
            # 툴 호출 관련 정보 로그
            logging.info(f"[CommonAgent] state_delta: {event.actions.state_delta}")
            logging.info(
                f"[CommonAgent] artifact_delta: {event.actions.artifact_delta}"
            )

            # 툴 호출 시도/결과 로그 (속성 존재 여부에 따라)
            if hasattr(event.actions, "tool_calls"):
                logging.info(f"[CommonAgent] tool_calls: {event.actions.tool_calls}")
            if hasattr(event.actions, "tool_results"):
                logging.info(
                    f"[CommonAgent] tool_results: {event.actions.tool_results}"
                )

            # 전체 actions 객체도 출력
            logging.info(f"[CommonAgent] actions(raw): {event.actions}")
            yield event
