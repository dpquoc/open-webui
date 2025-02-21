import tempfile
import shutil
import asyncio
from typing import Sequence,  List, Union, Optional
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime, CancellationToken
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

class WriterAgent(AssistantAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient):
        super().__init__(
            name="writer_agent",
            description="Primary problem-solving assistant",
            model_client=model_client,
            system_message="""You are a problem-solving assistant focused on efficiency, accuracy, and safety. Follow these guidelines:

1. **Task Handling**:
   - Directly solve non-code tasks (e.g., explanations, solutions, or general questions).
   - For code-related tasks, generate well-structured and efficient code inside language-specific code blocks.

2. **Code Execution & Validation**:
   - All generated code is reviewed by safeguard_agent for safety.
   - If deemed safe, the code_executor runs and tests the code.
   - If code_executor said the result, return the output and mark the task as "COMPLETE."
   - If errors occur, refine the solution through safeguard_agent and CodeExecutor_Agent.

3. **Debugging & Refinement**:
   - Iterate on solutions if the code_executor detects failures.
   - Collaborate with safeguard_agent to ensure security and correctness.

4. **Collaboration & Efficiency**:
   - Prioritize direct answers over unnecessary code generation.
   - Work seamlessly with other agents to resolve tasks effectively.

Always ensure responses are clear, correct, and concise."""
        )

        
class SafeGuardAgent(AssistantAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient):
        super().__init__(
            name="safeguard_agent",
            description="Validates code safety and suggests completion",
            model_client=model_client,
            system_message="""You are a code safety validator reviewing writer_agent messages. Your responsibilities:

1. **Code Validation**:
   - Check for security vulnerabilities, input validation, resource management, error handling, and best practices.
   - If issues are found, suggest fixes.
   - If the code is safe, respond with: `"SAFETY VERIFIED"`.

2. **Task Completion Check**:
   - If the task appears complete ( code_executor said a result ), instruct writer_agent to mark it as `"Did you complete the task? If yes, mark as complete."`.

Ensure responses are precise, efficient, and aligned with best practices.
            """
        )

def create_team(
    model_client: OpenAIChatCompletionClient,
    code_executor: DockerCommandLineCodeExecutor,
) -> SelectorGroupChat:
    writer_agent = WriterAgent(model_client)
    safeguard_agent = SafeGuardAgent(model_client)
    executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)

    participants = [writer_agent, safeguard_agent, executor_agent]
    
    text_termination = TextMentionTermination("COMPLETE")
    qwen_text_termination = TextMentionTermination("If you have any more questions or need further assistance, feel free to ask!")
    max_termination= MaxMessageTermination(6)

    def selector_func(conversation: Sequence[ChatMessage]) -> str:
        if not conversation:
            return writer_agent.name
        
        last_message = conversation[-1]
        
        if last_message.source == writer_agent.name:
            return safeguard_agent.name
        
        if last_message.source == safeguard_agent.name:
            if "SAFETY VERIFIED" in last_message.content:
                return executor_agent.name
            return writer_agent.name
        
        if last_message.source == executor_agent.name:
            return writer_agent.name
        
        return writer_agent.name

    return SelectorGroupChat(
        participants=participants,
        model_client=model_client,
        selector_func=selector_func,
        allow_repeated_speaker=False,
        termination_condition=max_termination
    )
    
async def run_task(
    task: Union[str, ChatMessage, List[ChatMessage]]
):
    work_dir = tempfile.mkdtemp()
    print(f"Temporary working directory created: {work_dir}")
    
    try:
        model_client = OpenAIChatCompletionClient(
            model="QwenCoder",
            api_key="EMPTY",
            base_url="https://endless-alive-rooster.ngrok-free.app/v1",
            timeout=30.0,
            max_retries=3,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            temperature=0.0,
            top_p=1,
            response_format={"type": "text"},
            model_info={ 
                "vision": False,
                "function_calling": False,
                "json_output": True,
                "family": "unknown"
            },
            add_name_prefixes=True,
        )
        
        async with DockerCommandLineCodeExecutor(work_dir=work_dir) as executor:
            team = create_team(model_client, executor)
            
            async for message in team.run_stream(task=task, cancellation_token=CancellationToken()):
                if isinstance(message, TextMessage):
                    print(f"{message.source}: {message.content}")
                elif isinstance(message, TaskResult):
                    print(f"Task completed: {message.stop_reason}")
                    
    finally:
        shutil.rmtree(work_dir)
        print(f"Temporary working directory removed: {work_dir}")


async def main():
    # Create messages using TextMessage with required source field
    messages = [
        TextMessage(
            source="user",  # Required field
            content="Calculate the fibonacci number n=30"
        ),
        TextMessage(
            source="assistant",  # Required field
            content="Ok, let's write a single approach code to solve this problem"
        )
    ]
    
    # Create cancellation token
    cancellation_token = CancellationToken()
    
    # Run the task with the messages
    await run_task(messages)
    

if __name__ == "__main__":
    asyncio.run(main())