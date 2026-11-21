import os
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API Error")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

JD = """
We are hiring a backend python developer.

Requirements:
-Strong python
-FastAPI or Django
-PostgreSQL
-Docker
-AWS
-REST APIs
-2+ years of experience
"""

RESUME = """
Name: Archit Jagadeb

Experience:
3 years as Software Developer

Skills:
Python, FastAPI, Docker, AWS, Git, MySQL

Project:
Built a food delivery backend using fastapi and mysql
deployed using docker
"""


def ask_llm(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def step1_res_extract(resume_text):
    system_prompt = """
    You are a professional HR assistant. Extract the skills from the candidate's resume provided.
    Only return the skills listed, no other information. Do not invent skills by yourself.
    """
    user_prompt = f"""
    Extract the skills from this resume:
    {resume_text}
    """
    return ask_llm(system_prompt, user_prompt)


def step2_jd_extract(jd_text):
    system_prompt = """
    You are a professional HR assistant. Extract the skills from the job description provided.
    Only return the skills listed, no other information. Do not invent skills by yourself.
    """
    user_prompt = f"""
    Extract the skills from this JD:
    {jd_text}
    """
    return ask_llm(system_prompt, user_prompt)


def step3_match(candidate, jd):
    system_prompt = """
    You are a professional HR assistant. Compare the skills of the candidate with the skills
    required in the JD and produce a final score between 1 and 100, plus a short verdict on
    whether the candidate is a good fit or not.
    Reply strictly in this format:
    Score: <number>
    Verdict: <one or two sentences>
    """
    user_prompt = f"""
    Compare and match the skills.
    JD skills: {jd}
    Candidate skills: {candidate}
    """
    return ask_llm(system_prompt, user_prompt)


candidate = step1_res_extract(RESUME)
sleep(2)  # remove if you don't hit rate limits
jd = step2_jd_extract(JD)
sleep(2)
result = step3_match(candidate, jd)
print(result)