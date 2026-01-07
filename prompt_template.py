from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
load_dotenv()

# Instantiate Model
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.7,
    max_completion_tokens=1000,
    verbose=True
)

#Prompt Template #1
prompt = ChatPromptTemplate.from_template("Tell me a joke above a {subject}")

#Prompt Template #2
prompt_2 = ChatPromptTemplate.from_messages([
    ("system","Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list.")
    ("user","{input}")
])

#Format the template #2
formatted_2 = prompt_2.format_messages(
    input = "World",
)

#Prompt Template #3
prompt_3 = PromptTemplate.from_template("Tell me a joke about {topic}}")

#Format the template
formatted = prompt_3.format(topic="cats")

#Create LLM Chain
chain = prompt | llm 

response = chain.invoke({"user":"World"})
print(response)