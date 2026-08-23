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
prompt="suggest a name for my company"

message_system = {
    "role": "system",
    "content": "you are a brand manager who suggest name for my drone company. name should be in one word. Suggest 2 names"
}

message = {
    "role": role,
    "content": prompt
}

messages = [message_system ,message]

reponse = client.chat.completions.create(model = model, messages=messages, temperature=0.5)
# print(reponse)

print("##############")

answer = reponse.choices[0].message.content
print(answer)