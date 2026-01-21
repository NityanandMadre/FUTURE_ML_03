import streamlit as st
from chatbot import chatbot_reply

st.set_page_config(page_title="Customer Support Chatbot")

st.title("🤖 Customer Support Chatbot")
st.write("Ask your support-related questions below")

user_input = st.text_input("Your message:")

if user_input:
    reply = chatbot_reply(user_input)
    st.success(reply)
