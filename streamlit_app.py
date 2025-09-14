# streamlit_app.py
import streamlit as st
from rag_pipeline import ask_question

st.title("🦺 OHS RAG Chatbot")

st.write("Ask any question about Occupational Health & Safety and get relevant answers from the OHS document.")

user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("Searching for answers..."):
        answer = ask_question(user_input)
    st.subheader("Answer:")
    st.write(answer)
