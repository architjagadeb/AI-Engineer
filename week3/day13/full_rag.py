import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2") #384 vector

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
 
if not my_api_key:
    raise ValueError("API Error")

client = Groq(api_key=my_api_key)
groqmodel = "openai/gpt-oss-120b"

documents = [
    "employees receive 24 fays of leave",
    "employess work from home on tuesday, wednesday, thursday",
    "monday and friday are optional work from home days",
    "employees receive Rs 3000 for gym reimbursement",
    "employees can claim Rs 2000 for home internet",
    "employees have 90 days notice period",
]

document_embedding = model.encode(documents)

def cosine_similarity(a, b):
    return np.dot(a, b) / (
    np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(qembedding):
    scores = []
    for i, document in enumerate (document_embedding):
        score = cosine_similarity(qembedding, document)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[0]

def ask_llm(question, context):
    # context = retrieve_info(question)
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
    response= client.chat.completions.create(model=groqmodel, messages=messages)
    answer=response.choices[0].message.content
    return answer

query = "how much vacations do i get?"
qembedding = model.encode(query)
score, context = retrieve(qembedding)
# print(score)
# print(context)
answer = ask_llm(query, context)
print(answer)