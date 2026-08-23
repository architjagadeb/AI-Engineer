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


from pydantic import BaseModel
class Ticket(BaseModel):
    name:str
    email:str
    issue:str
    address:str
    number:int

schema = Ticket.model_json_schema()
response_format ={
    "type": "json_object"
}
system_prompt=f"""
Extract the information strictly bsaed on this schema and give a json output {schema}
"""
message_system ={
    "role": "user",
    "content": system_prompt
}

text = "Hello my name is Archit. My Iphone is not working at all. My email is abc@gmail.com. My address is Delhi. My contact number is 3455464"
prompt=f"""
This is customer feedback. Extract details from this. {text}
"""
message = {
    "role": role,
    "content": prompt
}

messages = [message_system, message]

response = client.chat.completions.create(model = model, messages=messages, response_format=response_format)
print(response)

print("##############")

answer = response.choices[0].message.content
print(answer) 

import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)

print(ticket.name)
print(ticket.number)
print(ticket.email)