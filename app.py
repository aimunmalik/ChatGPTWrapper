import streamlit as st
import os
from openai import OpenAI
import io
import PyPDF2
import pdfplumber
import docx
from google.cloud import vision
from google.oauth2 import service_account
import json
import fitz  # PyMuPDF for PDF to image conversion

# Safe API key handling
def get_api_key():
    # Try Streamlit secrets first
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        if api_key and api_key.strip():
            return api_key
    except KeyError:
        pass
    except Exception:
        pass
    
    # Try environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.strip():
        return api_key
    
    st.error("❌ OpenAI API Key Required. Please add it to Streamlit Cloud secrets or environment variables.")
    st.info("💡 In Streamlit Cloud, go to App settings → Secrets and add: OPENAI_API_KEY = 'your_key_here'")
    st.stop()

# Initialize Google Vision client
def get_vision_client():
    try:
        # Try to get credentials from Streamlit secrets
        if "GOOGLE_CLOUD_CREDENTIALS" in st.secrets:
            credentials_dict = dict(st.secrets["GOOGLE_CLOUD_CREDENTIALS"])
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            return vision.ImageAnnotatorClient(credentials=credentials)
        
        # Try to get from environment variable
        creds_json = os.getenv("GOOGLE_CLOUD_CREDENTIALS")
        if creds_json:
            credentials_dict = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            return vision.ImageAnnotatorClient(credentials=credentials)
        
        # Try default credentials (if running on Google Cloud)
        return vision.ImageAnnotatorClient()
        
    except Exception:
        return None

# Initialize OpenAI client
try:
    client = OpenAI(api_key=get_api_key())
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.stop()

# Initialize Vision client (optional)
vision_client = get_vision_client()

# Streamlit UI
st.set_page_config(page_title="ANNA AI", page_icon="💬")
st.title("💬 Chat with ANNA AI")

# Display logo in sidebar
import base64
from PIL import Image

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return None

logo_base64 = get_base64_image("anna_logo.png")

with st.sidebar:
    if logo_base64:
        st.markdown(
            f"""
            <div style='text-align: center; padding-bottom: 10px;'>
                <img src='data:image/png;base64,{logo_base64}' width='120'>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("**ANNA AI**")

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
        st.rerun()

# Initialize message history and attachments
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "attachments" not in st.session_state:
    st.session_state.attachments = []

# Display previous messages
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "system":
        continue
    elif msg["role"] == "user":
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

# Google Vision OCR function
def extract_text_with_vision(file_bytes):
    """Extract text from PDF using Google Vision OCR"""
    if not vision_client:
        return "[Google Vision not available]"
    
    try:
        # Convert PDF to images using PyMuPDF
        doc = fitz.open("pdf", file_bytes)
        extracted_text = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
            img_bytes = pix.tobytes("png")
            
            # Use Google Vision to extract text
            image = vision.Image(content=img_bytes)
            response = vision_client.text_detection(image=image)
            
            if response.text_annotations:
                page_text = response.text_annotations[0].description
                extracted_text += f"\n--- Page {page_num + 1} (OCR) ---\n"
                extracted_text += page_text + "\n"
            else:
                extracted_text += f"\n--- Page {page_num + 1} (OCR) ---\n"
                extracted_text += "[No text detected on this page]\n"
            
            # Check for errors
            if response.error.message:
                extracted_text += f"[OCR Error: {response.error.message}]\n"
        
        doc.close()
        
        if extracted_text.strip():
            return f"[Google Vision OCR extracted {len(extracted_text)} characters]\n\n{extracted_text}"
        else:
            return "[Google Vision OCR: No text found in document]"
            
    except Exception as e:
        return f"[Google Vision OCR failed: {str(e)}]"

# Enhanced PDF extraction function with Vision fallback
def extract_text_from_pdf(file_bytes):
    try:
        import pdfplumber
        
        text = ""
        debug_info = ""
        
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            debug_info += f"PDF has {total_pages} page(s). "
            
            for page_num, page in enumerate(pdf.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                
                # Extract tables first (this is where pdfplumber shines)
                tables = page.extract_tables()
                if tables:
                    debug_info += f"Found {len(tables)} table(s) on page {page_num + 1}. "
                    for table_num, table in enumerate(tables):
                        text += f"\n** Table {table_num + 1} **\n"
                        for row in table:
                            if row and any(cell for cell in row if cell):  # Skip empty rows
                                # Clean and format row
                                clean_row = [str(cell).strip() if cell else "" for cell in row]
                                text += " | ".join(clean_row) + "\n"
                        text += "\n"
                
                # Extract regular text
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    # Remove text that's already in tables to avoid duplication
                    text += f"\n** Text Content **\n{page_text.strip()}\n"
                
                # If no content found
                if not tables and (not page_text or not page_text.strip()):
                    text += "[No readable content found on this page - may be images/scanned content]\n"
        
        total_chars = len(text.strip())
        debug_info += f"Extracted {total_chars} characters total."
        
        # If extraction was poor, try Google Vision as fallback
        if total_chars < 500:  # Threshold for "poor" extraction
            debug_info += " Attempting Google Vision OCR fallback..."
            
            if vision_client:
                vision_text = extract_text_with_vision(file_bytes)
                if vision_text and not vision_text.startswith("[Google Vision OCR failed"):
                    return f"[{debug_info} Using Google Vision OCR fallback.]\n\n{vision_text}"
            
            return f"[{debug_info} PDF appears to contain mostly images/scanned content.]"
        
        return f"[{debug_info}]\n\n{text.strip()}"
        
    except ImportError:
        # Fallback to PyPDF2 if pdfplumber not installed
        return extract_text_from_pdf_fallback(file_bytes)
    except Exception as e:
        # If pdfplumber fails, try Google Vision as fallback
        debug_info += f"pdfplumber failed: {str(e)}. "
        
        if vision_client:
            debug_info += "Attempting Google Vision OCR fallback..."
            vision_text = extract_text_with_vision(file_bytes)
            if vision_text and not vision_text.startswith("[Google Vision OCR failed"):
                return f"[{debug_info}]\n\n{vision_text}"
        
        return f"[{debug_info} All extraction methods failed.]"

# Fallback function using PyPDF2
def extract_text_from_pdf_fallback(file_bytes):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        # If PyPDF2 extraction is poor, try Google Vision
        if len(text.strip()) < 100 and vision_client:
            vision_text = extract_text_with_vision(file_bytes)
            if vision_text and not vision_text.startswith("[Google Vision OCR failed"):
                return f"[PyPDF2 extraction was poor, using Google Vision OCR fallback]\n\n{vision_text}"
        
        return text if text.strip() else "[PyPDF2 fallback: No text extracted]"
    except Exception as e:
        # Final fallback to Google Vision
        if vision_client:
            vision_text = extract_text_with_vision(file_bytes)
            if vision_text and not vision_text.startswith("[Google Vision OCR failed"):
                return f"[PyPDF2 failed, using Google Vision OCR]\n\n{vision_text}"
        
        return f"[All extraction methods failed: {str(e)}]"

def extract_text_from_docx(file_bytes):
    text = ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            if para.text.strip():  # Only add non-empty paragraphs
                text += para.text.strip() + "\n\n"
        
        if not text.strip():
            return "[Word document appears to be empty or contains only images/tables]"
            
        return text.strip()
    except Exception as e:
        return f"[Could not extract text from Word document. Error: {str(e)}]"

def extract_text_from_txt(file_bytes):
    try:
        # Try UTF-8 first
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            # Try other common encodings
            return file_bytes.decode("latin-1")
        except Exception as e:
            return f"[Could not decode text file. Error: {str(e)}]"

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
        for fname, text in extracted_texts:
            max_chars = 3000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...[truncated]..."
            user_message["content"] += f"\n\n[Content of attached file '{fname}']:\n{text}\n"
        file_names = ", ".join([att["filename"] for att in attachments])
        user_message["content"] += f"\n\n[User attached files: {file_names}]"
    
    st.session_state.messages.append(user_message)
    st.session_state.attachments.append(attachments)

    # Prepare messages for OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()
    except Exception as e:
        st.error(f"Error communicating with OpenAI: {str(e)}")

#run with streamlit run app.py