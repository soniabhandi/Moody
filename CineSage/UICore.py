import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Movie Info Extractor", page_icon="🎬", layout="centered")

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
        """,
        ),
        (
            "human",
            """
        Extract information from this movie paragraph:

        {movie_paragraph}

        Return the extracted information as valid JSON.
        """,
        ),
    ]
)

st.markdown('<div class="page-title">🎬 Movie Info Extractor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Paste a paragraph about a movie and get structured details back.</div>',
    unsafe_allow_html=True,
)

movie = st.text_area("Give your paragraph:", height=220, placeholder="Paste the movie paragraph here…")

if st.button("Extract", type="primary"):
    if not movie.strip():
        st.warning("Please paste a paragraph first.")
    else:
        with st.spinner("Extracting…"):
            final_prompt = prompt.invoke({"movie_paragraph": movie})
            response = model.invoke(final_prompt)
        st.subheader("Result")
        st.code(response.content, language="json")