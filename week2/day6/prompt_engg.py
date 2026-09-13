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

def llm_ans(prompt):

    message = {
        "role": "user",
        "content": prompt
    }

    messages = [message]
    response = client.chat.completions.create(model=model, messages=messages)
    ans = response.choices[0].message.content
    return ans

bad_prompt = '''
Act as a support assistant at a mobile/laptop company
You have to classify customer's issue in a category
Categories are billing, technical and return
if issue doesnot match with any categories, print fuck you
This is the user complaint:
i want to do a world tour
'''

print(llm_ans(bad_prompt))