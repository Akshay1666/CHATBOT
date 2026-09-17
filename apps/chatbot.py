from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="QuickAnswer",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# LLM + Tools
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    streaming=True
)

search = GoogleSerperAPIWrapper()
tools = [search.run]


# -----------------------------
# Memory
# -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# Agent
# -----------------------------
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt=(
        "You are an amazing AI agent and can search on Google as well."
    )
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Header */
    .quick-header {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 15px 0;
    }

    .quick-icon {
        font-size: 42px;
    }

    .quick-title {
        font-size: 32px;
        font-weight: 700;
        color: white;
        margin: 0;
    }

    .quick-subtitle {
        color: #9ca3af;
        font-size: 15px;
        margin-top: 3px;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 15px;
    }

    /* Input box */
    .stChatInput {
        border-radius: 15px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.markdown("## 🤖 QuickAnswer")

    st.caption("Your AI assistant with Google Search")

    st.divider()

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    if st.button("🔍 Search Enabled", use_container_width=True):
        st.info("Google Search is available to the AI agent.")

    st.divider()

    st.markdown("### ⚙️ Settings")

    st.checkbox(
        "Enable Google Search",
        value=True
    )

    st.divider()

    st.caption("Powered by Groq + LangGraph + Streamlit")


# -----------------------------
# Header
# -----------------------------
# st.markdown("""
# <div class="quick-header">

#     <div class="quick-icon">🤖</div>

#     <div>
#         <div class="quick-title">QuickAnswer</div>
#         <div class="quick-subtitle">
#             Answers at the speed of thought ⚡
#         </div>
#     </div>

# </div>
# """, unsafe_allow_html=True)


# st.divider()


# -----------------------------
# Display Chat History
# -----------------------------
for message in st.session_state.history:

    role = message["role"]
    content = message["content"]

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)

    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)


# -----------------------------
# Chat Input
# -----------------------------
query = st.chat_input(
    "💬 Ask anything..."
)


# -----------------------------
# Process Query
# -----------------------------
if query:

    # User message
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    st.session_state.history.append({
        "role": "user",
        "content": query
    })


    # Agent response
    response = agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        },
        stream_mode="messages"
    )


    # AI message
    with st.chat_message("assistant", avatar="🤖"):

        space = st.empty()

        message = ""

        for chunk in response:

            # Make sure chunk contains content
            if chunk[0].content:

                message += chunk[0].content

                space.markdown(message)


    st.session_state.history.append({
        "role": "ai",
        "content": message
    })