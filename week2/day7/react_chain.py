import os
import re
from time import sleep
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API Error")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

# tools
def get_product_price(product):
    if product == 'iphone 18':
        return 1000
    elif product == 'iphone 17':
        return 500
    else:
        return 0

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "calc error!"

tools = {
    "get_product_price": get_product_price,
    "calculator": calculator
}

system_prompt = '''
You are a shopping assistant.
You have these tools:

get_product_price(product)
calculator(expression)
important:
Call tool exactly like these examples;

Action: get_product_price("iphone 18")
Action: calculator("1000 - 500")

never write:
get_product_price(product = 'iphone 18')

never write:
calculator(expression = "1000 - 500")

follow these rules:
1. decide what you need to do next
2. call only one tool at a time
3. after writing an action, stop immediately
4. never guess or invent a tool result
5. wait until you recieve an observation
6. then decide your next action
7. when task is complete, give the final answer

Format:
thought: what you need to do
action: tool_name(argument)

When finished:
final answer: your answer
'''

def run_agent(question):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for step in range(5):
        print("\n-----------")
        print("STEP", step + 1)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        ans = response.choices[0].message.content
        print(ans)

        # agent has finished
        if "final answer:" in ans.lower():
            break

        # find action
        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            ans
        )

        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            tool_input = tool_input.strip()
            tool_input = tool_input.strip('"')

            # run the tool
            if tool_name in tools:
                tool = tools[tool_name]
                observation = tool(tool_input)
            else:
                observation = "tool not found"
        else:
            observation = "no action found in response"

        print(
            "Observation: ",
            observation
        )

        # add llm response to memory
        messages.append({
            "role": "assistant",
            "content": ans
        })

        # give tool result back to llm
        messages.append({
            "role": "user",
            "content": "observation: " + str(observation)
        })
        sleep(5)

prompt = '''
I have 2000 rupees. WHat is the price of iphone 18? and how much i will be left with after buying it
'''

run_agent(prompt)