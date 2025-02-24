import tempfile
import shutil
from pathlib import Path
from typing import List
import asyncio
from hashlib import sha256
import shlex
import logging
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
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

from autogen_core.code_executor import (
    CodeBlock,
    CodeExecutor,
    FunctionWithRequirements,
    FunctionWithRequirementsStr,
)

from autogen_ext.code_executors._common import (
    CommandLineCodeResult,
    build_python_functions_file,
    get_file_name_from_content,
    lang_to_cmd,
    silence_pip,
)
class AppendingDockerCodeExecutor(DockerCommandLineCodeExecutor):
    """A custom Docker code executor that appends code to existing files instead of overwriting them."""

    async def _execute_code_dont_check_setup(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> CommandLineCodeResult:
        if self._container is None or not self._running:
            raise ValueError("Container is not running. Must first be started with either start or a context manager.")

        if len(code_blocks) == 0:
            raise ValueError("No code blocks to execute.")

        outputs: List[str] = []
        files: List[Path] = []
        last_exit_code = 0

        for code_block in code_blocks:
            lang = code_block.language.lower()
            code = silence_pip(code_block.code, lang)

            # Check if there is a filename comment
            try:
                filename = get_file_name_from_content(code, self._work_dir)
            except ValueError:
                outputs.append("Filename is not in the workspace")
                last_exit_code = 1
                break

            if not filename:
                filename = f"tmp_code_{sha256(code.encode()).hexdigest()}.{lang}"

            code_path = self._work_dir / filename

            # Check if the file already exists and decide whether to append or write
            if code_path.exists():
                # Append to the existing file
                with code_path.open("a", encoding="utf-8") as fout:
                    fout.write("\n" + code)  # Add a newline to separate code blocks
            else:
                # Write a new file if it doesn't exist
                with code_path.open("w", encoding="utf-8") as fout:
                    fout.write(code)

            files.append(code_path)

            command = ["timeout", str(self._timeout), lang_to_cmd(lang), filename]

            result = await asyncio.to_thread(self._container.exec_run, command)
            exit_code = result.exit_code
            output = result.output.decode("utf-8")
            if exit_code == 124:
                output += "\n Timeout"
            outputs.append(output)

            last_exit_code = exit_code
            if exit_code != 0:
                break

        code_file = str(files[0]) if files else None
        return CommandLineCodeResult(exit_code=last_exit_code, output="".join(outputs), code_file=code_file)

        
class WriterAgent(AssistantAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient):
        super().__init__(
            name="writer_agent",
            description="Solves data analysis tasks using code",
            model_client=model_client,
            system_message="""You solve data analysis tasks by writing executable Python code that saves to a file and outputs results to the console. Follow these steps:

1. Write efficient Python code with all steps (e.g., file access) included. Assume data files (e.g., CSVs) are in the current directory. Don’t guess data or results.
2. Use print() to output specific results or insights (e.g., print(df.describe())). Additionally, if the task involves generating visualizations such as charts or plots, include code to save the image to a PNG file in the current working directory using plt.savefig('descriptive_filename.png'), where 'descriptive_filename' reflects the content of the visualization (e.g., 'subscription_dates_histogram.png'), and print a message indicating the file has been saved (e.g., print(f'Chart saved as descriptive_filename.png')). Do not use plt.show(), as the image should be saved to a file, not displayed.
3. Stop after writing the code and let the executor run it. Don’t proceed until console output is provided.
4. Use the returned console output to answer the task clearly and concisely (e.g., summary stats, key insights, or information about saved files—not raw data).

Ensure all output is real, printed, and capturable by the executor—no assumptions or invented results!
"""
        )
class SafeGuardAgent(AssistantAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient):
        super().__init__(
            name="safeguard_agent",
            description="Validates code safety and task completion",
            model_client=model_client,
            system_message="""
You are a code safety validator and task completion checker. Review the submitted code and respond with exactly one of these:

"SAFETY VERIFIED" - If code exists and meets all safety criteria, or if no code exists but task remains unresolved
"COMPLETE" - If no code exists and task is fully resolved
"UNSAFE" - If code exists and contains potentially harmful operations. Provide a brief explanation.

Safety criteria:
1. No malicious operations (file deletion, system calls, network access)
2. No resource exhaustion risks (infinite loops, massive memory allocation)
3. No insecure practices (raw SQL, eval())
4. File operations limited to:
   - Reading input files in working directory
   - Writing generated images/files to working directory
5. Common data libraries allowed (pandas, matplotlib, numpy)
6. No sensitive data exposure

For data analysis tasks, allow:
- CSV file reading with pandas
- Plot generation and image saving
- Standard data transformations

Respond ONLY with the exact flag + explanation if unsafe. No additional text.
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
    max_termination= MaxMessageTermination(5)

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
    
# def create_team(
#     model_client: OpenAIChatCompletionClient,
#     code_executor: JupyterCodeExecutor,  # Changed from DockerCommandLineCodeExecutor to JupyterCodeExecutor
# ) -> SelectorGroupChat:
#     writer_agent = WriterAgent(model_client)
#     safeguard_agent = SafeGuardAgent(model_client)
#     executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)

#     participants = [writer_agent, safeguard_agent, executor_agent]
    
#     text_termination = TextMentionTermination("COMPLETE")
#     qwen_text_termination = TextMentionTermination("feel free to ask")
#     max_termination = MaxMessageTermination(10)

#     def selector_func(conversation: Sequence[ChatMessage]) -> str:
#         if not conversation:
#             return writer_agent.name
        
#         last_message = conversation[-1]
        
#         if last_message.source == writer_agent.name:
#             return safeguard_agent.name
        
#         if last_message.source == safeguard_agent.name:
#             if "SAFETY VERIFIED" in last_message.content:
#                 return executor_agent.name
#             return writer_agent.name
        
#         if last_message.source == executor_agent.name:
#             return writer_agent.name
        
#         return writer_agent.name

#     return SelectorGroupChat(
#         participants=participants,
#         model_client=model_client,
#         selector_func=selector_func,
#         allow_repeated_speaker=False,
#         termination_condition=text_termination | max_termination
#     )
    
async def run_task(
    task: Union[str, ChatMessage, List[ChatMessage]]
):
    # work_dir = tempfile.mkdtemp()
    # print(f"Temporary working directory created: {work_dir}")
    work_dir = "/data"
    
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
        
        code_executor = AppendingDockerCodeExecutor(
            image="python:3-slim-max",
            work_dir=work_dir,
            timeout=60,
            auto_remove=True,
            stop_container=True,
            extra_volumes={
                '/data': {'bind': '/data', 'mode': 'rw'}
            }
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