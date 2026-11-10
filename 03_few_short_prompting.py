import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
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


# Define few shot examples
few_shot_examples = [
  {"movie": "Inception", "review": "A mind-bending thriller that keeps you on the edge of your seat.", "genre": "Genre Found -> Science Fiction"},
  {"movie": "The Godfather", "review": "An epic tale of family, power, and betrayal.", "genre": "Genre Found -> Crime"},
  {"movie": "The Shawshank Redemption", "review": "A story of hope and friendship that transcends the walls of a prison", "genre": "Genre Found -> Drama"}
]

# Template for one example
example_template = PromptTemplate(
    input_variables=["movie", "review", "genre"],
    template="Movie: {movie}\nReview: {review}\n{genre}\n"
)

print("----Few shot prompting demo------")
print(example_template)

#Create a FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_template,
    examples=few_shot_examples,
    suffix= "Movie: {movie} \nGenre:",
    input_variables=["movie"]
)

query = { "movie": "Pulp Fiction"}

# Without few-shot prompting
prompt_template = PromptTemplate(
    input_variables=["movie"],
    template="Movie: {movie} \nGive me the review and genre of the movie."
)
filled_prompt = prompt_template.format(movie=query["movie"])
print(f"Filled prompt without few-shot prompting: {filled_prompt}")
result = llm.invoke(filled_prompt)
print (f"Response without few-shot prompting: {result.content}")
print("-"*100)

# With few-shot prompting
filled_prompt_few_shot = few_shot_prompt.format(**query)
print(f"Filled prompt with few-shot prompting: {filled_prompt_few_shot}")
result_few_shot = llm.invoke(filled_prompt_few_shot)
print (f"Response with few-shot prompting: {result_few_shot.content}")
print("-"*100)