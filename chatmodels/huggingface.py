from dotenv import load_dotenv
import os
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash-1017"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the name of the planet that we live")

print(response.content)