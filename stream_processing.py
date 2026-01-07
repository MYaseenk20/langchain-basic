from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.7,
    max_completion_tokens=1000,
    verbose=True
)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{text}") 
    ]
)

chain = chat_template | llm

for chunk in chain.stream({
    "input_language": "English",
    "output_language": "French",
    "text": "Hello, how are you?"
}):
    print(chunk.content, end="", flush=True)
