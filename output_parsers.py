from langchain.chat_models import init_chat_model
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

load_dotenv()


parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("wire a short poem about {topic}")


llm = init_chat_model(model="gpt-4o-mini", temperature=0)

chain = prompt | llm | parser

response = chain.invoke({"topic": "nature"})

print(response)

print(type(response))


# JsonOutputParser example
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with 'name' and 'age' for: {description}"
)

chain = prompt | llm | parser

result = chain.invoke({"description": "A 31-year-old developer named Alex"})

print(result)

# PydanticOutputParser example
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# class Person(BaseModel):
#     name: str = Field(description="The person's name")
#     age: int = Field(description="The person's age")
#     occupation: str = Field(description="The person's occupation")


# parser = PydanticOutputParser(pydantic_object=Person)

# prompt = ChatPromptTemplate.from_template(
#     "Return a JSON object with 'name', 'age', and 'occupation' for: {description}"
# ).partial(format_instructions=parser.get_format_instructions())

# chain = prompt | llm | parser

# result2 = chain.invoke({"description": "A 41-year-old Engineer named Man"})

# print(result2)


# Structured Output
class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")


# Bind the schema to the model
structured_model = llm.with_structured_output(MovieReview)

result = structured_model.invoke("Review: Inception is a mind-binding thriller. 9/10")

print(result)
