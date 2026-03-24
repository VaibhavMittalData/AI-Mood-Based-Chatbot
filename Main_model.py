import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Super App", page_icon="🤖")

st.title("🚀 Unified AI App (Mood + RAG)")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

mood_choice = st.sidebar.radio(
    "Choose Mood (Optional):",
    ["😎 Normal", "😡 Angry", "😂 Funny", "😢 Sad"],
    horizontal=False
)

st.sidebar.markdown("---")
st.sidebar.header("📄 Document Upload (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload PDF/TXT for RAG", type=["pdf", "txt"])

# ---------------- LLM & EMBEDDINGS ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Determine System Prompt based on Mood
if mood_choice == "😡 Angry":
    base_system_prompt = "You are an angry AI. Respond aggressively."
elif mood_choice == "😂 Funny":
    base_system_prompt = "You are a funny AI. Crack jokes."
elif mood_choice == "😢 Sad":
    base_system_prompt = "You are a sad AI. Respond emotionally."
else:
    base_system_prompt = "You are a helpful and polite AI assistant."

# Update SystemMessage if mood changes
if "mode" not in st.session_state or st.session_state.mode != base_system_prompt:
    st.session_state.mode = base_system_prompt
    # Filter out old system messages
    messages = [msg for msg in st.session_state.messages if not isinstance(msg, SystemMessage)]
    # Insert new system message at the beginning
    st.session_state.messages = [SystemMessage(content=base_system_prompt)] + messages

# ---------------- PROCESS UPLOAD ----------------
if uploaded_file and st.session_state.current_file != uploaded_file.name:
    with st.spinner("Processing document..."):
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Initialize Chroma for the uploaded file
        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )
        st.session_state.current_file = uploaded_file.name
        st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")

elif not uploaded_file and st.session_state.current_file:
    # Optional: Clear RAG state if file is removed
    st.session_state.vectorstore = None
    st.session_state.current_file = None
    st.sidebar.info("Removed document context.")

# ---------------- CHAT INTERFACE ----------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user input
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append(HumanMessage(content=user_input))

    # Determine if we should use RAG
    if st.session_state.vectorstore is not None:
        retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})
        docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Combine system prompt + context for RAG
        rag_prompt = f"""{base_system_prompt}
        
Context from document:
{context}

Answer the user's question primarily based on the context. If the answer is not in the context, say 'Not in document' but you can still respond using your persona."""
        
        # Temporarily use this RAG prompt as the system prompt for this invocation
        temp_messages = [SystemMessage(content=rag_prompt)] + [msg for msg in st.session_state.messages if not isinstance(msg, SystemMessage)]
        response = llm.invoke(temp_messages)
    else:
        # Standard chat invocation
        response = llm.invoke(st.session_state.messages)

    # Display AI response
    st.session_state.messages.append(AIMessage(content=response.content))
    with st.chat_message("assistant"):
        st.write(response.content)

# Clear button in sidebar
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages = [SystemMessage(content=base_system_prompt)]
    st.session_state.vectorstore = None
    st.session_state.current_file = None
    st.rerun()