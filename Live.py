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
            raise ValueError("API_Agents_Key1 not found in environment variables. Please check your .env file.")
        
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
            return f"Error generating response: {str(e)}", False

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
    st.subheader("📊 RAG Model Status")
    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        st.success("✓ RAG Model: Active")
        st.caption("The chatbot will use uploaded documents to answer questions.")
    else:
        st.warning("○ RAG Model: Inactive")
        st.caption("Upload a file to enable RAG-enhanced responses.")
    
    # List uploaded files
    uploaded_files_list = list(UPLOADED_FILES_DIR.glob("*"))
    if uploaded_files_list:
        st.markdown("**Uploaded Files:**")
        for file in uploaded_files_list:
            st.text(f"• {file.name}")

# Right column: Chat Interface
with col2:
    st.subheader("🤖 Chat with HR Assistant")
    
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

**Supported File Types:**
- 📊 Excel: .xlsx, .xls
- 📄 PDF: .pdf
- 📝 Word: .docx
- 📋 CSV: .csv
- 📃 Text: .txt, .md, and other text files

**Example Questions:**
- "What information is in the onboarding file?"
- "How many employees are listed?"
- "Show me details about John Doe"
- "What is the average bill rate?"
""")
