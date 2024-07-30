import os

# from litellm import completion
from openai import OpenAI

api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(
    api_key='test',  # token-abc123
    base_url='http://localhost:8000/v1',
)

model = 'Meta-Llama-3.1-70B-Instruct'
# model = 'Meta-Llama-3.1-70B-Instruct-Backup'
system_message = """Summarize the given action within 5 words in 5 DIFFERENT ways, separated by a newline character. NOTE: only respond with the summaries, don't give headers such as \'here are the summaries\' etc."""
processed_prompt = "Calculate the age difference between the current US President and the current leader of Russia by subtracting Vladimir Putin's age from Joe Biden's age."
messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': processed_prompt},
]


# messages = [{'role': 'user', 'content': 'what llm are you'}]
# response = completion(
#     model="meta-llama/Meta-Llama-3.1-70B-Instruct",
#     messages=messages,
#     api_key='sk-1234',
#     api_base='http://localhost:8000/v1/',
#     temperature=0.2,
#     max_tokens=80,
# )
def call_chat_api(prompt, client, model, system_message=None, **kwargs):
    completion = client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    content = completion.choices[0].message.content.split('\n')
    return content


print(call_chat_api(processed_prompt, client, model))
