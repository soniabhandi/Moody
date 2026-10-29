from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from typing import List,Optional

load_dotenv()

class Movie(BaseModel):
    title:str
    release_year : Optional[int]
    genre : List[str]
    director : Optional[str]
    cast : List[str]
    rating :Optional[float]
    summary :str

parser = PydanticOutputParser(pydantic_object=Movie)

model = ChatGroq(
    model ="openai/gpt-oss-120b",
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages(
    [
       (
        "system",
        """
        Extract movie info from the paragraph
        {format_instructions}
        """
    ),
    (
        "human",
        """
        {movie_paragraph}
        """
    )  
    ]
)

movie = input("Give your paragraph : " )

final_prompt = prompt.invoke(
    {"movie_paragraph": movie,"format_instructions":parser.get_format_instructions}
)

response = model.invoke(final_prompt)

print(response.content)

