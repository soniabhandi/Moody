import streamlit as st
import os



from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


def get_groq_api_key():
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if api_key:
            return api_key
    except (FileNotFoundError, RuntimeError):
        pass

    return os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Chatworld", page_icon="💬", layout="centered")

# ------------------------------------------------------------------ roles ---
ROLES = {
    "Funny": {
        "emoji": "😂",
        "tagline": "Jokes first, answers second.",
        "system": "You are a funny AI Agent",
        "accent": "#E08A00",
        "tint": "#FFF6E3",
        "greeting": "Say something. I promise to make it weirder.",
    },
    "Sad": {
        "emoji": "😢",
        "tagline": "Helpful, but heavy-hearted.",
        "system": "You are a sad AI Agent",
        "accent": "#3F6FB5",
        "tint": "#E8F0FB",
        "greeting": "Oh... you wanted to talk. Go ahead, I guess.",
    },
    "Angry": {
        "emoji": "😡",
        "tagline": "Answers everything. Annoyed about it.",
        "system": "You are an angry AI Agent",
        "accent": "#C93A2B",
        "tint": "#FDEBE8",
        "greeting": "What now? Fine. Ask your question.",
    },
}


# @st.cache_resource
# def get_model():
#     return ChatGroq(
#         model="openai/gpt-oss-120b",
#         temperature=0.5,
#     )


@st.cache_resource
def get_model():
    api_key = get_groq_api_key()

    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it to Streamlit Secrets or your local .env file.")
        st.stop()

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5,
        api_key=api_key,
    )



model = get_model()

# ------------------------------------------------------------------ state ---
if "role" not in st.session_state:
    st.session_state.role = "Funny"
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(ROLES["Funny"]["system"])]


def switch_role():
    """Changing the role starts a fresh chat with that role's system message."""
    new_role = st.session_state.role_picker
    st.session_state.role = new_role
    st.session_state.messages = [SystemMessage(ROLES[new_role]["system"])]


def clear_chat():
    st.session_state.messages = [SystemMessage(ROLES[st.session_state.role]["system"])]


role = st.session_state.role
cfg = ROLES[role]

# ------------------------------------------------------------------ style ---
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Nunito', system-ui, sans-serif; }
.stApp { background: __TINT__; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2rem; max-width: 760px; }

section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E4E4E7; }
.side-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 0.2rem; }
.side-sub { color: #6B7280; font-size: 0.88rem; margin-bottom: 1.2rem; }

.role-head { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.4rem; }
.role-badge {
    width: 3.4rem; height: 3.4rem; border-radius: 50%;
    background: __ACCENT__; display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem; flex-shrink: 0;
}
.role-name { font-size: 1.7rem; font-weight: 800; line-height: 1.1; color: #1F2430; }
.role-tag { color: #5B6270; font-size: 0.95rem; }
.divider { height: 3px; width: 100%; background: __ACCENT__; opacity: 0.25; border-radius: 3px; margin: 1rem 0 1.4rem 0; }

.empty { color: #5B6270; text-align: center; padding: 2.5rem 1rem; font-size: 1.05rem; }
.empty b { color: __ACCENT__; }

[data-testid="stChatMessage"] {
    background: #FFFFFF; border: 1px solid rgba(0,0,0,0.07);
    border-radius: 16px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
}
[data-testid="stChatInput"] textarea:focus { box-shadow: 0 0 0 2px __ACCENT__; }

.stButton > button {
    border-radius: 10px; font-weight: 700; border: 1px solid #D4D4D8;
}
.stButton > button:hover { border-color: __ACCENT__; color: __ACCENT__; }
</style>
"""
st.markdown(
    css.replace("__ACCENT__", cfg["accent"]).replace("__TINT__", cfg["tint"]),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown('<div class="side-title">Chatworld</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Pick who you want to talk to.</div>', unsafe_allow_html=True)

    st.radio(
        "Role",
        list(ROLES.keys()),
        index=list(ROLES.keys()).index(role),
        key="role_picker",
        on_change=switch_role,
        format_func=lambda r: f"{ROLES[r]['emoji']}  {r} AI",
    )
    st.caption("Switching roles starts a new conversation.")

    st.divider()
    st.button("Clear chat", on_click=clear_chat, use_container_width=True)

# ------------------------------------------------------------------- main ---
st.markdown(
    f"""
<div class="role-head">
  <div class="role-badge">{cfg['emoji']}</div>
  <div>
    <div class="role-name">{role} AI</div>
    <div class="role-tag">{cfg['tagline']}</div>
  </div>
</div>
<div class="divider"></div>
""",
    unsafe_allow_html=True,
)

history = [m for m in st.session_state.messages if not isinstance(m, SystemMessage)]

if not history:
    st.markdown(f'<div class="empty"><b>{cfg["greeting"]}</b></div>', unsafe_allow_html=True)

for m in history:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.markdown(m.content)
    elif isinstance(m, AIMessage):
        with st.chat_message("assistant", avatar=cfg["emoji"]):
            st.markdown(m.content)

if prompt := st.chat_input(f"Message the {role.lower()} AI"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # with st.chat_message("assistant", avatar=cfg["emoji"]):
    #     with st.spinner("Thinking…"):
    #         chat_model = model.invoke(st.session_state.messages)
    #     st.markdown(chat_model.content)

    with st.chat_message("assistant", avatar=cfg["emoji"]):
        with st.spinner("Thinking…"):
            try:
                chat_model = model.invoke(st.session_state.messages)
                st.markdown(chat_model.content)
                st.session_state.messages.append(
                    AIMessage(content=chat_model.content)
                )
            except Exception:
                st.error("Sorry, I couldn't get a response. Please try again.")

    st.session_state.messages.append(AIMessage(content=chat_model.content))
    st.rerun()