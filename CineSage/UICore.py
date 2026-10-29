
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from typing import List, Optional
import json

load_dotenv()

st.set_page_config(
    page_title="Movie Info Extractor",
    page_icon="🎬",
    layout="centered"
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Nunito', system-ui, sans-serif; }
.stApp { background: #F5F5F7; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.5rem; max-width: 780px; }
.page-title { font-size: 2rem; font-weight: 800; color: #1F2430; margin-bottom: 0.2rem; }
.page-sub { color: #5B6270; margin-bottom: 1.5rem; }
.stButton > button[kind="primary"] {
    background: #B3261E; border-color: #B3261E; border-radius: 10px; font-weight: 700;
}
.stButton > button[kind="primary"]:hover { background: #8F1E18; border-color: #8F1E18; }
.stTextArea textarea { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


# Pydantic schema
class Movie(BaseModel):
    title: Optional[str] = None
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str
    language: Optional[str] = None
    country: Optional[str] = None
    main_characters: List[str] = []
    themes: List[str] = []
    setting: Optional[str] = None
    tone: List[str] = []
    target_audience: List[str] = []
    content_warnings: List[str] = []
    keywords: List[str] = []
    missing_information: List[str] = []


# Pydantic output parser
parser = PydanticOutputParser(pydantic_object=Movie)


@st.cache_resource
def get_model():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5,
    )


model = get_model()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert movie information extraction assistant.

            Extract useful movie information from the supplied paragraph.

            Rules:
            1. Extract information only from the supplied paragraph.
            2. Never invent facts.
            3. Use null for missing optional fields.
            4. Use empty lists for missing list fields.
            5. Distinguish actors from characters where possible.
            6. Infer genres, themes, tone, and audience only when
               reasonably supported by the paragraph.
            7. Write a clear summary in 2-3 sentences.
            8. List missing information in missing_information.
            9. Follow the schema exactly.

            {format_instructions}
            """
        ),
        (
            "human",
            """
            Extract information from this movie paragraph:

            {movie_paragraph}
            """
        ),
    ]
)


st.markdown(
    '<div class="page-title">🎬 Movie Info Extractor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-sub">Paste a paragraph about a movie and get structured details back.</div>',
    unsafe_allow_html=True,
)

movie = st.text_area(
    "Give your paragraph:",
    height=220,
    placeholder="Paste the movie paragraph here…"
)


if st.button("Extract", type="primary"):
    if not movie.strip():
        st.warning("Please paste a paragraph first.")
    else:
        with st.spinner("Extracting…"):
            try:
                final_prompt = prompt.invoke({
                    "movie_paragraph": movie,
                    "format_instructions":
                        parser.get_format_instructions()
                })

                response = model.invoke(final_prompt)

                # Parse and validate the LLM response
                parsed_result = parser.parse(response.content)

                # Convert Pydantic object to a dictionary
                result_json = parsed_result.model_dump()

                st.subheader("Result")

                # Display formatted JSON in Streamlit
                st.json(result_json)

            except Exception as e:
                st.error(f"Could not extract movie information: {e}")
