"""
LangChain Core Concepts - LCEL and Runnables
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model


load_dotenv()


def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "Your are a helper assistant. Answer in one sentence: {question}"
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Compose with pipe operator
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}")


def demo_batch_exectution():
    """Demonstrate batch execution for multiple inputs"""
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Batch - run with multiple inputs
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "What do you like?"},
    ]

    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")


def demo_streaming():
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    print("Streaming output")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)

    print()


def exercise_first_chain():

    prompt = ChatPromptTemplate.from_template(
        "You are a marketing specialist and you goal is to create a marketing tagline for a product named '{product}' targeting '{audience}'"
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    result = chain.invoke({"product": "news website", "audience": "student"})

    print(result)


def demo_schema_inspection():
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


def new_way():
    model = init_chat_model("gpt-40-mini", temperature=0.7, max_tokens=1500)

    print(model)


if __name__ == "__main__":
    new_way()
