import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
 
if not my_api_key:
    raise ValueError("API Error")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"
role = "user"

prompt1 = "Hi"
prompt2 = "give some details on time travel"
prompt3 = "write a paragraph on delhi"

prompts = [prompt1, prompt2, prompt3]
for prompt in prompts:
    message = {
    "role": role,
    "content": prompt
    }
    messages = [message]
    response = client.chat.completions.create(model = model, messages=messages, max_tokens=50)
    usage =response.usage
    print(f"Prompt: {prompt} --> your tokens: {usage.prompt_tokens} completion tokens: {usage.completion_tokens} total tokens: {usage.total_tokens} Finish reason: {response.choices[0].finish_reason}")

# prompt="do you know narendra modi"

# reponse = client.chat.completions.create(model = model, messages=messages)
# print(reponse)

# print("##############")

# answer = reponse.choices[0].message.content
# print(answer)