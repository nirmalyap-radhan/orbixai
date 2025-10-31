"""
AI Chatbot Application using LangChain and Groq API
Developed by Nirmalya Pradhan | © 2025
"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="AI Chatbot | Nirmalya Pradhan",
    page_icon="🤖",
    layout="wide"
)

# --- App Header ---
st.markdown("""
    <h2 style='text-align: center;'>OrbiX AI</h2>
    <p style='text-align: center; color: gray;'>Built by Nirmalya Sonu using LangChain and Groq API</p>
    <hr>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
    "Select Model",
    [
        "llama-3.1-8b-instant",      # Fast, lightweight, good for chatbots
                      # Powerful mixture-of-experts model
    ],
    index=0
    )


    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Create LangChain Chat Chain ---
@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant powered by Groq. Answer questions clearly and concisely."),
        ("user", "{question}")
    ])

    return prompt | llm | StrOutputParser()

chain = get_chain(GROQ_API_KEY, model_name)

# --- Check for API Key ---
if not GROQ_API_KEY:
    st.error("🚫 No GROQ_API_KEY found. Please set it in your .env file.")
elif not chain:
    st.error("🚫 Could not initialize Groq chain. Check your API key validity.")
else:
    # --- Display Past Messages ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # --- Chat Input ---
    if question := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- Example Prompts ---
st.markdown("""
    <hr>
    <h4 style='text-align: center;'>💡 Try Asking Me</h4>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("- Build an AI bot?")
    st.markdown("- Explain LangChain in one line")
with col2:
    st.markdown("- Suggest AI project ideas")
    st.markdown("- Write a Groq haiku")

# --- Footer ---
st.markdown("""
    <hr>
    <p style='text-align: center; color: gray;'>
        Made with ❤️ by <strong>Nirmalya Pradhan</strong><br>
        © 2025 Nirmalya Pradhan | All rights reserved.
    </p>
""", unsafe_allow_html=True)

