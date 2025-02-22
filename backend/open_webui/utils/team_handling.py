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
   - For tasks that do not require code (e.g., explanations, general questions), provide a direct solution or answer. Once you have fully addressed the task, say 'COMPLETE' to end the session.
   - For tasks that require code (e.g., calculations, data processing), generate well-structured and efficient code inside language-specific code blocks. Include print statements to output the results.

2. **Code Execution & Validation**:
   - Your generated code will be sent to a safeguard agent to confirm its safety.
   - If the code is safe, it will be executed by an executor agent, and the results will be provided back to you.
   - Use these execution results to formulate your final response to the user's task.
   - After you have incorporated the results and fully addressed the user's task, say 'COMPLETE' to end the session.

3. **Important Notes**:
   - Do not log or interpret the code results yourself; rely on the exact results provided by the executor agent.
   - Ensure that you say 'COMPLETE' only after the task is fully resolved, meaning you have provided a complete answer or solution to the user's question or problem.
   
"""
        )

        
class SafeGuardAgent(AssistantAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient):
        super().__init__(
            name="safeguard_agent",
            description="Validates code safety",
            model_client=model_client,
            system_message="""
You are a code safety validator. Review the code provided. If it's safe, respond with 'SAFETY VERIFIED'. If it's not safe, briefly explain why.
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
    qwen_text_termination = TextMentionTermination("feel free to ask")
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
        termination_condition= text_termination | max_termination
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
        
        # Use the existing container 'autogen' running python:3-slim
        code_executor = DockerCommandLineCodeExecutor(
            image="python:3-slim-max",     # Still specify the image for consistency
            # container_name="autogen",        # The Executor will create a Docker Contain with this name , make sure it's not duplicaated with existed one
            work_dir=work_dir,
            timeout=60,                      # Adjust as needed
            auto_remove=True,            # Don’t remove since it’s pre-existing
            stop_container=True          # Don’t stop since it’s already running
        )
        
        async with code_executor as executor:
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