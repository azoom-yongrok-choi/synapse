from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm


class ParsingAgent(LlmAgent):
    def __init__(self, ctx, tools):
        super().__init__(
            name="parsing_agent",
            model=LiteLlm(model="openai/gpt-4o"),
            instruction="""
You are a universal natural language parser.

Your task is to:
1. Analyze the user's input and choose the appropriate schema tool.
2. Call the selected tool to get the schema (keys and types).
3. Use the schema to parse the user input into a JSON object.

Guidelines:
- If a field is optional and not present in input, set it to null.
- Always return all required keys.
- Return only the final JSON result.
""",
            output_key="json_result",
            tools=tools,
        )

    async def run_parsing(self, ctx, user_input: str) -> dict:
        tool_list = [
            f"- {tool.name}: {tool.description or 'No description'}"
            for tool in ctx.available_tools
        ]
        tools = "\n".join(tool_list)
        tool_prompt = f"""
User input: "{user_input}"

Select one of the following tools based on user intent:
{tools}

Respond with tool name only.
"""
        tool_name = await self.run(tool_prompt)

        tool_result = await ctx.call_tool(tool_name.strip(), {"input": user_input})
        schema_str = tool_result["content"][0]["text"]

        final_prompt = f"""
Schema: {schema_str}
User input: "{user_input}"

Now parse the input into a JSON object using the schema.
"""
        json_result = await self.run(final_prompt)
        return json_result
