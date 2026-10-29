from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()



model = ChatGroq(
    model ="openai/gpt-oss-120b",
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages(
    [
       (
        "system",
        """
        You are an expert movie information extraction assistant.

        Your task is to extract useful movie information from the
        movie paragraph supplied by the user.

        Follow these rules:
        1. Extract information only from the supplied paragraph.
        2. Never invent a movie title, actor, director, date, or fact.
        3. If a single-value field is missing, return null.
        4. If a list field is missing, return an empty list.
        5. Distinguish actors from the characters they portray.
        6. Infer genres, themes, tone, and audience only when
           reasonably supported by the paragraph.
        7. Write a clear summary in 2-3 sentences.
        8. Keep the output concise and useful for a movie database.
        9. Return only data matching the requested output schema.

        Extract these fields:
        - movie_name
        - release_year
        - genres
        - director
        - cast (actor and character)
        - language
        - country
        - summary
        - main_characters
        - themes
        - setting
        - tone
        - target_audience
        - content_warnings
        - keywords
        - missing_information

        Output format:
        Return a valid JSON object with exactly these keys.
        Do not include Markdown fences or explanatory text.
        """
    ),
    (
        "human",
        """
        Extract information from this movie paragraph:

        {movie_paragraph}

        Return the extracted information as valid JSON.
        """
    )  
    ]
)

movie = input("Give your paragraph : " )

final_prompt = prompt.invoke(
    {"movie_paragraph": movie}
)

response = model.invoke(final_prompt)

print(response.content)

