import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 RAG Chatbot with File Upload")

# -------------------------------
# Upload File
# -------------------------------
uploaded_file = st.file_uploader("Upload a file (PDF or TXT)", type=["pdf", "txt"])

# -------------------------------
# Embedding Model
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Load & Process File
# -------------------------------
if uploaded_file:
    file_path = os.path.join("temp", uploaded_file.name)

    os.makedirs("temp", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    st.success("✅ File processed and stored!")

else:
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

# -------------------------------
# Retriever
# -------------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# -------------------------------
# LLM
# -------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# -------------------------------
# Prompt
# -------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If not found, say: "I could not find the answer in the document."
"""),
    ("human", """Context:
{context}

Question:
{question}
""")
])

# -------------------------------
# Chat UI
# -------------------------------
query = st.text_input("Ask your question:")

if query:
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    st.subheader("💬 Answer")
    st.write(response.content)