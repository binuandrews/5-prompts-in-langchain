import json

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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

direct_template = PromptTemplate(
    input_variables=["question"],
    template="{question}?"
)

cot_template = PromptTemplate(
    input_variables=["question"],
    template="Let's think step by step before answering \n {question} \nNow give the final concise answer:"
)

questions = ["If a bat and a ball cost $1.10 in total, and the bat costs $1.00 more than the ball, how much does the ball cost?",
             "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?"]

def run_and_print(formatted_prompt: str):
  try:
    response = llm.invoke(formatted_prompt)
    if hasattr(response, "content"):
      return response.content
    else:
      return str(response)
  except Exception as e:
    return f"Error invoking LLM: {e}"

for i, question in enumerate(questions, start=1):
    # print(f"{i}--------------------{question}")
    print("_" * 100)
    # Direct/zero-shot prompting
    direct_prompt = direct_template.format(question=question)
    print(f"Direct prompt: {direct_prompt}")
    print(f"Response for direct prompt: {run_and_print(direct_prompt)}")
    print("=" * 100)


    # Chain of thought prompting
    cot_prompt = cot_template.format(question=question)
    print(f"Chain of Thought prompt: {cot_prompt}")
    print(f"Response for Chain of Thought prompt: {run_and_print(cot_prompt)}")
    print("_" * 100)


