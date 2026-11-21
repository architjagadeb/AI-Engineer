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

#step1
knowledge_base= {
    "age":"the age of archit is 19",
    "net worth":"the net worth of archit is 0"
}

#step2 retrieval
def retrieve_info(question):
    question=question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net worth" in question:
        return knowledge_base["net worth"]
    else:
        return None
def ask_llm(question):
    context = retrieve_info(question)
    sys_prompt = f"""answer in one line only. answer based on context. do not hallucinate. context: {context}"""
    system_messages={
        "role":"system",
        "content": sys_prompt
    }
    message={

        "role":"user",
        "content": question
    }
    messages = [system_messages, message]
    response= client.chat.completions.create(model=model, messages=messages)
    answer=response.choices[0].message.content
    return answer


question = "what is archit jagadeb's age"
print(ask_llm(question))