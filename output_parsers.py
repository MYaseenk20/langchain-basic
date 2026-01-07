from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,CommaSeparatedListOutputParser,JsonOutputParser
from pydantic import BaseModel, Field
# Instantiate Model
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.7,
    max_completion_tokens=1000,
    verbose=True
)

#Output Parser Function String
def call_string_output_parser():
   
    # Prompt Template
   prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Tell me joke about following subject"),
            ("human", "{input}")
        ]
    )
   
   parser = StrOutputParser()

    # Create LLM Chain
   chain = prompt | llm | parser
   
   return chain.invoke({"input":"dog"})

#Output Parser Function List
def call_list_output_parser():

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
            ("human", "{input}")
        ]
    )

    parser = CommaSeparatedListOutputParser()

    # Create LLM Chain
    chain = prompt | llm | parser

    return chain.invoke({"input": "happy"})

def call_json_output_parser():

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extract information from the following phrase.\nFormatting Instructions: {format_instructions}"),
            ("human", "{phrase}")
        ]
    )

    class FoodRecipe(BaseModel):
        recipe : str = Field(description="the name of the recipe")
        ingredients: list = Field(description="ingredients")

    parser = JsonOutputParser(pydantic_object=FoodRecipe)

    chain = prompt | llm | parser

    return chain.invoke({
        "phrase" : "The ingredients for a Margherita pizza are tomatoes, onions, cheese, basil",
        "format_instructions" : parser.get_format_instructions()
    })

print(call_json_output_parser())