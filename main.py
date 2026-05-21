from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

from langchain_core import __version__ as core_version

# from langgraph import __version__ as lg_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

lg_version = version("langgraph")


print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")

    llm_anthropic = ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929", temperature=0
    )
    response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatAnthropic: {response_anthropic}")


if __name__ == "__main__":
    main()
