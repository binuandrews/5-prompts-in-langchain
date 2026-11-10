import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
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


chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a creative assistant who writes in a friendly tone"),
    ("user", "Write a short greeting message for a friend named {name} who loves {hobby}.")
])

filled_chat_prompt = chat_prompt_template.format_messages(name="Alice", hobby="painting")

print("----Chat Prompt template demo------")
print(f"The Filled chat prompt is::: {filled_chat_prompt}")
print("-"*100)
print(f"Sending prompt to LLM ({type(llm).__name__}) for response...")
print("-"*100)
response = llm.invoke(filled_chat_prompt)
print(f"Printing the response below *****\n\n{response.content}")
print("-"*100)