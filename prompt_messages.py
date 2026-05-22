from dotenv import load_dotenv


from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

load_dotenv()


def demo_basic_templates():
    # simple = ChatPromptTemplate.from_template(
    #     "Tell me a {adjective} joke about {topic}"
    # )

    # messages = simple.format_messages(adjective="funny", topic="beef")

    # print(messages)

    multi = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}",
            ),
            ("human", "Translate the following text: {text}"),
        ]
    )

    messages = multi.format_messages(
        input_language="English", output_language="French", text="I love programming."
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response = model.invoke(messages)

    print(response.content)


from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


messages = [
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there! How can I assist you today?"),
    SystemMessage(content="This is a system message."),
    # ToolMessage(content="Tool executed successfully"),
    # ChatMessage(content="This is a general chat message", role="user"),
]


def demo_few_shot():

    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )

    fewshot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt, examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give the opposite of each word."),
            fewshot_prompt,
            ("human", "{input}"),
        ]
    )

    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    response = model.invoke(final_prompt.format_messages(input="happy"))

    print(response.content)


def demo_prompt_composition():

    persona = ChatPromptTemplate.from_messages(
        ["system", "you are a {role}, your tone is {tone}"]
    )

    task = ChatPromptTemplate.from_messages(messages=["human", "{task}"])

    full_prompt = persona + task

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = full_prompt | model

    response = chain.invoke(
        {"role": "teacher", "tone": "patient", "task": "tell me about your students"}
    )

    print(response.content)

    response2 = chain.invoke(
        {"role": "scientist", "tone": "patient", "task": "tell me about your research"}
    )

    print(response2.content)


if __name__ == "__main__":
    demo_prompt_composition()
