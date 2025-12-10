# HR AI Chatbot with RAG Integration - Streamlit Interface
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Import RAG functionality
from Loading_Document import build_faiss_index, query_faiss

# Load environment variables
load_dotenv()

# Configuration
st.set_page_config(page_title="HR AI Chatbot", page_icon="💼", layout="wide")

# Data directory setup
DATA_DIR = Path(r"c:\Users\samso\Downloads\Python\Python Agents-\data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR = DATA_DIR / "faiss_index"
UPLOADED_FILES_DIR = DATA_DIR / "uploaded_files"
UPLOADED_FILES_DIR.mkdir(parents=True, exist_ok=True)

class HRChatbot:
    def __init__(self):
        # Initialize client with API key from environment
        self.api_key = os.getenv("API_Agents_Key1")
        if self.api_key:
            self.api_key = self.api_key.strip()
        if not self.api_key:
            raise ValueError("❌ API_Agents_Key1 not found in environment variables. Please check your .env file.")
        
        # Validate API key format
        if not self.api_key.startswith('sk-or-v1-'):
            raise ValueError("❌ Invalid OpenRouter API key format. Key should start with 'sk-or-v1-'")
        
        self.system_prompt = """ You are a HR assistant with document management capabilities. 
        
Your key responsibilities:
1. Answer HR-related questions thoroughly and professionally (2-5 sentences)
2. Help users upload and onboard documents by guiding them to the file upload section
3. When documents are uploaded, use the provided context to give accurate, context-aware answers
4. If users ask about uploading files, direct them to use the file uploader on the left side of the interface

Be friendly, helpful, and proactive in assisting with document management and HR queries."""
        
        # initializing the model
        self.llm = ChatOpenAI(
            model_name="google/gemini-2.0-flash-001", 
            temperature=0.5, 
            api_key=self.api_key, 
            base_url="https://openrouter.ai/api/v1",
            model_kwargs={
                "extra_headers": {
                    "HTTP-Referer": "https://localhost",
                    "X-Title": "HR Chatbot"
                }
            }
        )

    def detect_upload_intent(self, user_input: str) -> bool:
        """Detect if user wants to upload/onboard a document"""
        upload_keywords = [
            'upload', 'onboard', 'add file', 'add document', 'submit file',
            'load file', 'import', 'upload file', 'attach', 'send file'
        ]
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in upload_keywords)
    
    def detect_view_file_intent(self, user_input: str) -> bool:
        """Detect if user wants to view uploaded files"""
        view_keywords = [
            'view file', 'see file', 'show file', 'preview', 'download file',
            'view document', 'see document', 'show document', 'view upload',
            'see uploaded', 'show uploaded', 'access file'
        ]
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in view_keywords)

    def get_response(self, user_input: str, history: list) -> tuple[str, bool]:
        """
        Get response from LLM, using RAG if available.
        Returns: (response_text, is_upload_request)
        """
        try:
            # Check for upload intent
            if self.detect_upload_intent(user_input):
                upload_guide = """I'd be happy to help you onboard a document! 📁
                
To upload a file, please use the **"Upload Onboarding Files"** section on the left side of this page.

**Steps:**
1. Click the "Browse files" or "Choose a file" button in the left panel
2. Select your document (Excel, PDF, Word, CSV, Text, etc.)
3. The file will be automatically processed and indexed in the RAG system
4. Once uploaded, you can ask me questions about the content!

**Supported file types:**
- 📊 Excel (.xlsx, .xls)
- 📄 PDF (.pdf)
- 📝 Word (.docx)
- 📋 CSV (.csv)
- 📃 Text (.txt, .md, and more)

Would you like help with anything else after uploading your file?"""
                return upload_guide, True
            
            # Check for view file intent
            if self.detect_view_file_intent(user_input):
                view_guide = """To view your uploaded files, check the **"📂 Uploaded Files"** section in the left panel! 📂

**What you can do:**
1. **Preview Files**: Click on any file name to expand and view its contents
   - Text/CSV files: See a preview of the content
   - Excel files: View the first 10 rows in a table
   - PDF/Word: Download to view locally

2. **Download Files**: Each file has a ⬇️ Download button to save it to your computer

3. **File Information**: See file size and format for each uploaded document

**Alternatively**, you can ask me questions about the content:
- "What's in the file?"
- "Show me the data from [filename]"
- "Summarize the uploaded document"

The files are stored and indexed, so I can answer questions about their content anytime!"""
                return view_guide, True
            
            # Check if FAISS index exists and use RAG
            if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
                # Use RAG-enhanced response
                rag_answer = query_faiss(user_input, str(FAISS_DIR), k=4, llm_temperature=0.5)
                return rag_answer, False
            else:
                # Use standard LLM without RAG
                messages = history + [HumanMessage(content=user_input)]
                response = self.llm.invoke(messages)
                return response.content, False
        except Exception as e:
            error_msg = str(e)
            
            # Provide user-friendly error messages
            if "401" in error_msg or "User not found" in error_msg:
                return """❌ **API Authentication Error**
                
Your OpenRouter API key appears to be invalid or expired.

**To fix this:**
1. Visit https://openrouter.ai/keys
2. Log in to your account
3. Create a new API key
4. Update the `API_Agents_Key1` in your `.env` file
5. Restart the Streamlit application

**Note:** Make sure you have credits in your OpenRouter account.""", False
            elif "api_key" in error_msg.lower():
                return f"❌ **API Key Error**: {error_msg}\n\nPlease check your OpenRouter API key in the .env file.", False
            else:
                return f"❌ **Error**: {error_msg}", False

# Initialize session state
if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = HRChatbot()
        st.session_state.history = [SystemMessage(content=st.session_state.chatbot.system_prompt)]
        st.session_state.chat_messages = []
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()

# Page Header
st.title("💼 HR AI Assistant")
st.markdown("---")

# Layout: Two columns
col1, col2 = st.columns([1, 2])

# Left column: File Upload Section
with col1:
    st.subheader("📁 Upload Onboarding Files")
    st.markdown("Upload documents in any format. Supported: **Excel** (.xlsx, .xls), **PDF** (.pdf), **Word** (.docx), **CSV** (.csv), **Text** (.txt, .md), and more. Files will be automatically processed and stored in the RAG model.")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=None,  # Accept all file types
        help="Upload any document file (Excel, PDF, Word, Text, CSV, etc.)"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file
            file_path = UPLOADED_FILES_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Build FAISS index
            with st.spinner("Processing file and building RAG index..."):
                build_faiss_index(
                    source_path=str(file_path),
                    persist_dir=str(FAISS_DIR),
                    chunk_size=300,
                    chunk_overlap=30
                )
            
            st.success(f"✅ File '{uploaded_file.name}' uploaded and indexed successfully!")
            st.info("💡 You can now ask questions about the uploaded data.")
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    st.markdown("---")
    
    # Display RAG status
    st.subheader("📊 Status")
    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        st.success("✓ RAG Model: Active")
    else:
        st.warning("○ RAG Model: Inactive")
    
    # API Key status
    api_key = os.getenv("API_Agents_Key1")
    if api_key and api_key.startswith('sk-or-v1-'):
        st.success("✓ API: Connected")
    else:
        st.error("✗ API: Invalid Key")
    
    st.markdown("---")
    
    # List uploaded files with preview option
    uploaded_files_list = list(UPLOADED_FILES_DIR.glob("*"))
    if uploaded_files_list:
        st.subheader("📂 Uploaded Files")
        for file in uploaded_files_list:
            with st.expander(f"📄 {file.name}"):
                # Show file info
                file_size = file.stat().st_size
                st.caption(f"Size: {file_size / 1024:.2f} KB")
                
                # Add download button
                with open(file, "rb") as f:
                    st.download_button(
                        label="⬇️ Download",
                        data=f.read(),
                        file_name=file.name,
                        mime="application/octet-stream"
                    )
                
                # Try to preview file content
                try:
                    if file.suffix.lower() in ['.txt', '.md', '.csv']:
                        st.text("Preview:")
                        content = file.read_text(encoding='utf-8', errors='ignore')
                        st.text_area("", value=content[:1000] + ("..." if len(content) > 1000 else ""), height=200, key=f"preview_{file.name}")
                    elif file.suffix.lower() in ['.xlsx', '.xls']:
                        st.text("Excel Preview (first 10 rows):")
                        import pandas as pd
                        df = pd.read_excel(file)
                        st.dataframe(df.head(10))
                    elif file.suffix.lower() == '.pdf':
                        st.info("PDF file uploaded. Ask me questions about its content!")
                    elif file.suffix.lower() == '.docx':
                        st.info("Word document uploaded. Ask me questions about its content!")
                    else:
                        st.info("File uploaded successfully. Ask me questions about its content!")
                except Exception as e:
                    st.warning(f"Cannot preview this file type. Use download to view locally.")
    else:
        st.caption("No files uploaded yet")

# Right column: Chat Interface
with col2:
    st.subheader("💬 Chat with HR Assistant")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask me anything about HR, onboarding, or your uploaded documents...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        st.session_state.history.append(HumanMessage(content=user_input))
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
        
        # Get bot response
        with st.spinner("Thinking..."):
            response, is_upload_request = st.session_state.chatbot.get_response(user_input, st.session_state.history)
        
        # Add assistant response to chat
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.session_state.history.append(AIMessage(content=response))
        
        # Display assistant response
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(response)
                
                # If it's an upload request, add a helpful highlight
                if is_upload_request:
                    st.info("👈 Look at the left panel to upload your file!")
        
        # Rerun to update chat display
        st.rerun()

# Footer with instructions
st.markdown("---")
st.markdown("""
### 📝 How to Use:
1. **Upload Files**: Use the left panel to upload any document (Excel, PDF, Word, CSV, Text, etc.)
2. **Ask Questions**: Type your questions in the chat box on the right
3. **RAG-Enhanced Responses**: When files are uploaded, the chatbot will use them to provide accurate, context-aware answers

**Example Questions:**
- "I want to upload a file" (for upload guidance)
- "What information is in the onboarding file?"
- "How many employees are listed?"
- "Show me details about John Egnore"
""")
