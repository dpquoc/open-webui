from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage, SystemMessage

# Define your custom client with the required parameters
custom_client = OpenAIChatCompletionClient(
    model="QwenCoder",
    api_key="EMPTY",
    base_url="https://endless-alive-rooster.ngrok-free.app/v1",
    timeout=30.0,
    max_retries=3,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    temperature=0.3,
    top_p=0.9,
    response_format={"type": "text"},
    model_info={ 
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": "unknown"
    },
    # max_tokens=150,  # Set the maximum number of tokens in the response
    # stop=["\n"],  # Specify stop sequences if needed
    add_name_prefixes=True, ## THIS IS REALLY IMPORTANT FOR ROLE PLAY
)

# Create messages using the proper message types
messages = [
    SystemMessage(content="You are a helpful assistant.", source="system"),
    UserMessage(content="Hi", source="user"),
    UserMessage(content="I'm AGENT", source="agent"),
    UserMessage(content="Nothing", source="user"),
    # UserMessage(content="Explain the concept of machine learning in simple terms.", source="user")
]

# Use async/await pattern for creating the response
async def get_response():
    result = await custom_client.create(messages)
    print(result)

# Run the async function
import asyncio
asyncio.run(get_response())