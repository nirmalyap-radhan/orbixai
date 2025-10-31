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

load_dotenv()  # Load environment variables from orbix.env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
st.set_page_config(
    page_title="AI Chatbot | Nirmalya Pradhan",
    page_icon="🤖",
    layout="wide"
)

# heading
st.markdown("""
    <h2 style='text-align: center;'>OrbiX AI</h2>
    <p style='text-align: center; color: gray;'>Built by Nirmalya Sonu using LangChain and Groq API</p>
    <hr>
""", unsafe_allow_html=True)

# sidebar 
with st.sidebar:
    st.header("Settings")

    # choose a model
    model_name = st.selectbox(
        "Select Model",
        ["gemma2-9b-it",],
        index=0
    )

    # clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# create message list if not already there
if "messages" not in st.session_state:
    st.session_state.messages = []

# chat model chain
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

# create the chain using the API key and model
chain = get_chain(GROQ_API_KEY, model_name)

# show error if API key is missing or wrong
if not chain:
    st.error("🚫 Invalid API key. Please check the hardcoded GROQ_API_KEY in the code.")
else:
    # show past chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # get user input and generate reply
    if question := st.chat_input("Ask me anything"):
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

# show example questions
st.markdown("""
    <hr>
    <h4 style='text-align: center;'>💡 Try Asking Me</h4>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("- Build an AI bot?")
    st.markdown("- LangChain in one line")
with col2:
    st.markdown("- AI project ideas?")
    st.markdown("- Groq haiku please")

# footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: gray;'>
        Made with ❤️ by <strong>Nirmalya Pradhan</strong><br>
        © 2025 Nirmalya Pradhan | All rights reserved.
    </p>
""", unsafe_allow_html=True)



