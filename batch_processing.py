from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# Instantiate Model
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.7,
    max_completion_tokens=1000,
    verbose=True
)

#Prompt Template
chat_template = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human","{text}")
])

#LCEL (Pipe operator) 
chain = chat_template | llm

inputs = [
    {"input_language":"English","output_language":"French","text":"Hello"},
    {"input_language": "English", "output_language": "Spanish", "text": "Goodbye"},
    {"input_language": "English", "output_language": "German", "text": "Thank you"}
]

response = chain.batch(inputs)

for result in response:
    print(result.content)