import json

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

with open("config.json", "r") as f:
  config = json.load(f)

provider = config["provider"]
if provider == "openai":
  llm = ChatOpenAI(
      model=config["openai"]["model"],
      temperature=config["openai"]["temperature"],
      max_tokens=config["openai"]["max_tokens"]
  )
elif provider == "gemini":
  llm = ChatGoogleGenerativeAI(
      model=config["gemini"]["model"],
      temperature=config["gemini"]["temperature"],
      max_tokens=config["gemini"]["max_tokens"]
  )
else:
  raise ValueError("Invalid provider in config.json. Must be 'openai' or 'gemini'")

# Define personas via system role

personas = {
  "Teacher": "You are a knowledgeable and patient teacher. You explain concepts clearly and provide examples to help students understand.",
  "Lawyer": "You are a skilled lawyer. You provide legal advice and explain complex legal concepts in a clear and concise manner.",
  "Coder": "You are a proficient programmer. You explains things usng clean, efficient code and explain technical concepts in a straightforward manner.",
  "Finance Expert": "You are a financial expert. You provide insights on investments, market trends, and financial planning in a clear and concise manner."
}

question = "Can you explain the concept of blockchain technology in simple terms?"
print(("--------Role/persona based prompting demo--------"))
print("-" * 100)
print(f"Question: {question}\n")

for role, persona_description in personas.items():
  chat_prompt =ChatPromptTemplate.from_messages([
    ("system", persona_description),
    ("user", "{question}")
  ])

  filled_prompt = chat_prompt.format_messages(question=question)

  # Print persona being tested
  print(f"Role: {role}, Filled Prompt: {filled_prompt}")
  print("_"*(10+len(role)))
  print("System Role Prompt:")
  print(persona_description)
  print("\n Model Response:")

  #Invoke Model
  response = llm.invoke(filled_prompt)
  print(response.content)
  print("-" * len(response.content)+"\n\n")