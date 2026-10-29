from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage

model = ChatGroq(
    model ="openai/gpt-oss-120b",
    temperature=0.5
)

messages=[
    SystemMessage("You are a funny AI Agent")
]

print("Welcome to chatworld, Type 0 to exit")

while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content = prompt))
    if prompt == "0":
        break
    chat_model = model.invoke(messages)
    messages.append(AIMessage(content = chat_model.content))
    print("Bot: ",chat_model.content)

print(messages)


