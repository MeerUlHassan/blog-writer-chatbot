import streamlit as st
from graph import BlogWriter
import os
import base64
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

st.set_page_config(page_title="Blog Writer ChatBot", page_icon="✍️")
st.title("✍️ AI Blog Writer ChatBot 🚀")
st.image("./media/cover2.jpg", use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey, your writing companion is ready—what’s the plan?"}]
    st.session_state.app = None
    st.session_state.chat_active = True

with st.sidebar:
    st.subheader("Generate professional, comprehensive, and visually appealing blogs on any topic!")
    st.info(
        " * This app uses the OpenAI API to generate text.\n\n"
        " * The app saves your writing in a well-organized PDF format.\n\n"
        " * Writing a blog may take some time, so please be patient (approximately 1-2 minutes)."
    )

def initialize_agents():
    st.secrets.get("OPENAI_API_KEY") == OPENAI_API_KEY
    blog_writer = BlogWriter().graph

    if len(OPENAI_API_KEY) < 1:
        st.error("OpenAI API key is missing. Please contact the administrator.")
        st.session_state.chat_active = True
    else:
        st.success("Agents successfully initialized")
        st.session_state.chat_active = False

    return blog_writer

with st.sidebar:
    if st.button("Initialize Agents", type="primary"):
        st.session_state.app = initialize_agents()
    st.divider()

app = st.session_state.app

def generate_response(topic):
    return app.invoke(input={"topic": topic})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if topic := st.chat_input(placeholder="Ask a question", disabled=st.session_state.chat_active):
    st.chat_message("user").markdown(topic)

    st.session_state.messages.append({"role": "user", "content": topic})
    with st.spinner("Thinking..."):
        response = generate_response(topic)

    with st.chat_message("assistant"):
        if "pdf_name" in response:
            with open(f"./{response['pdf_name']}", "rb") as file:
                file_bytes = file.read()
                b64 = base64.b64encode(file_bytes).decode()
            href = f"<a href='data:application/octet-stream;base64,{b64}' download='{response['pdf_name']}'>Click here to download the PDF</a>"

            st.markdown(f"{response['response']}: {href}", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": f"{response['response']}: {href}"})
        else:
            st.markdown(response["response"])
            st.session_state.messages.append({"role": "assistant", "content": response["response"]})
