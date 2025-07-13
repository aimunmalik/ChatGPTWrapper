import streamlit as st
import os
# from dotenv import load_dotenv
# load_dotenv()
from openai import OpenAI
import io
import PyPDF2
import docx
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit UI
st.set_page_config(page_title="ANNA AI", page_icon="💬")
st.title("💬 Chat with ANNA AI")

# Display your logo in sidebar
import base64
import io
from PIL import Image

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

logo_base64 = get_base64_image("anna_logo.png")

with st.sidebar:
    st.markdown(
        f"""
        <div style='text-align: center; padding-bottom: 10px;'>
            <img src='data:image/png;base64,{logo_base64}' width='120'>
        </div>
        """,
        unsafe_allow_html=True
    )
# Sidebar: Model selection and Clear Chat button
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Choose GPT model",
        options=["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
        index=0
    )
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        st.session_state.attachments = []
        st.experimental_rerun()

# Initialize message history and attachments
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "attachments" not in st.session_state:
    st.session_state.attachments = []

# Display previous messages with nice formatting
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "system":
        continue  # Optionally skip displaying system message
    elif msg["role"] == "user":
        # Check for attachments for this message
        attachment_info = ""
        if "attachments" in msg and msg["attachments"]:
            for att in msg["attachments"]:
                attachment_info += f'<div style="font-size:90%;color:#1976d2;">📎 {att["filename"]}</div>'
        st.markdown(
            f'<div style="background-color:#e1f5fe; padding:10px; border-radius:8px; margin-bottom:5px;"><b>🧑 You:</b> {msg["content"]}{attachment_info}</div>',
            unsafe_allow_html=True,
        )
    elif msg["role"] == "assistant":
        st.markdown(
            f'<div style="background-color:#f1f8e9; padding:10px; border-radius:8px; margin-bottom:5px;"><b>🤖 Assistant:</b> {msg["content"]}</div>',
            unsafe_allow_html=True,
        )

# Helper functions to extract text from files
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        text = "[Could not extract text from PDF]"
    return text

def extract_text_from_docx(file_bytes):
    text = ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        text = "[Could not extract text from DOCX]"
    return text

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode("utf-8")
    except Exception:
        return "[Could not extract text from TXT]"

def extract_text_from_file(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file.read())
    elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        return extract_text_from_docx(file.read())
    elif file.type.startswith("text/"):
        return extract_text_from_txt(file.read())
    else:
        return "[Unsupported file type for text extraction]"

# Get user input and file attachments
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:")
    uploaded_files = st.file_uploader("Attach files (optional)", accept_multiple_files=True)
    submitted = st.form_submit_button("Send")

if submitted and (user_input or uploaded_files):
    # Save attachments to session state for this message
    attachments = []
    extracted_texts = []
    if uploaded_files:
        for file in uploaded_files:
            file.seek(0)
            file_bytes = file.read()
            # Extract text for supported file types
            file.seek(0)
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file_bytes)
            elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                text = extract_text_from_docx(file_bytes)
            elif file.type.startswith("text/"):
                text = extract_text_from_txt(file_bytes)
            else:
                text = "[Unsupported file type for text extraction]"
            extracted_texts.append((file.name, text))
            attachments.append({
                "filename": file.name,
                "type": file.type,
                "size": len(file_bytes),
                "content": file_bytes
            })
    # Add user message with attachments
    user_message = {"role": "user", "content": user_input}
    if attachments:
        user_message["attachments"] = [{"filename": att["filename"], "type": att["type"], "size": att["size"]} for att in attachments]
        # Append extracted text to the message content for LLM input
        for fname, text in extracted_texts:
            # Truncate text if too long for context window
            max_chars = 3000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...[truncated]..."
            user_message["content"] += f"\n\n[Content of attached file '{fname}']:\n{text}\n"
        file_names = ", ".join([att["filename"] for att in attachments])
        user_message["content"] += f"\n\n[User attached files: {file_names}]"
    st.session_state.messages.append(user_message)
    st.session_state.attachments.append(attachments)

    # Prepare messages for OpenAI (attachments are not sent, but extracted text is included)
    response = client.chat.completions.create(
        model=model,
        messages=st.session_state.messages
    )

    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

## Run with streamlit run app.py