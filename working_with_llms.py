from dotenv import load_dotenv


from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import os


load_dotenv()


def demo_init_chat_model():
    chat_model = init_chat_model(
        model_provider="openai",
        model="gpt-40-mini",
        temperature=0.7,
        max_tokens=1500,
        streaming=True,
    )

    return chat_model


def demo_model_comparison():
    prompt = "Explain recursion in one sentence"

    models = {
        "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini", temperature=0.7, streaming=False
        ),
        "gpt-4o": init_chat_model(model="gpt-4o", temperature=0.7, streaming=False),
    }

    # add anthropic model if available
    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude-sonnet-4-5-20250929"] = init_chat_model(
            model="claude-sonnet-4-5-20250929",
            model_provider="anthropic",
            temperature=0.7,
            streaming=False,
        )

    print(f"Prompt: {prompt}\n")
    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}")


def demo_message():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=False)

    # using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate"),
        HumanMessage(content="What's the weather like today?"),
    ]

    # print("Using message objects:")
    # print(f"Messages: {messages[0]} | {messages[1]}")
    response = model.invoke(messages)

    messages.append(response)
    messages.append(HumanMessage(content="What about tomorrow"))

    print(messages)


def exercise_multi_model():

    def get_responses(question: str, model_names: list[str]) -> dict[str, str]:
        responses = {}
        for model_name in model_names:
            model = init_chat_model(model=model_name, temperature=0.7, streaming=False)
            response = model.invoke(question)
            responses[model_name] = response.content

        return responses

    results = get_responses("what is AI?", ["gpt-4o-mini", "gpt-4o"])

    for model, answer in results.items():
        print(f"Response from {model}: {answer}\n")


if __name__ == "__main__":
    exercise_multi_model()
