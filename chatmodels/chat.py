from dotenv import load_dotenv

load_dotenv()

# from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

# model = init_chat_model(
#     "openai/gpt-oss-120b",
#     model_provider="groq",
#     temperature=0.3
# )

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3
) 

response = model.invoke("what is the capital of USA")

print(response.content)
