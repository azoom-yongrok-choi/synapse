# back_office_agent/main_async.py
import asyncio
from back_office_agent.agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types
import logging

logging.basicConfig(level=logging.INFO, force=True)


async def async_main():
    logging.info("[MAIN] async_main() called")
    session_service = InMemorySessionService()
    artifacts_service = InMemoryArtifactService()
    session = session_service.create_session(
        state={}, app_name="mcp_app", user_id="user1"
    )

    runner = Runner(
        app_name="mcp_app",
        agent=root_agent,
        artifact_service=artifacts_service,
        session_service=session_service,
    )

    query = "list files in the tests folder"
    content = types.Content(role="user", parts=[types.Part(text=query)])

    print("Running agent...")
    try:
        events_async = runner.run_async(
            session_id=session.id, user_id=session.user_id, new_message=content
        )
        async for event in events_async:
            print(f"Event received: {event}")
    except Exception as e:
        print(f"[System Error] {str(e)}")


if __name__ == "__main__":
    asyncio.run(async_main())
