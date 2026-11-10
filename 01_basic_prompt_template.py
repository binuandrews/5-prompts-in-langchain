import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import json

os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "NONE"


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

prompt_template = PromptTemplate(
    input_variables=["topic", "style"],
    template="Write a short poem about {topic} in {style}"
)

# Dynamically fill in variables in the prompt
filled_prompt = prompt_template.format(topic="Machine Learning", style="gen-z accent")

print("----Prompt template demo------")
print(f"THe Filled prompt is::: {filled_prompt}")
print("-"*100)

response = llm.invoke(filled_prompt)
print(f"Printing the response below *****\n\n{response.content}")