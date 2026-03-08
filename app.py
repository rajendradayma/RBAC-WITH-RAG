import os
import json
import uuid
import streamlit as st
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

# LangChain and Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
FAISS_INDEX_PATH = "./local_faiss_index"
MANIFEST_PATH = "./local_faiss_manifest.json"
UPLOAD_DIR = "./uploaded_pdfs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Data Models ---
@dataclass
class Document:
    file_id: str
    name: str
    read_access: List[str]
    modified_time: Optional[str] = None
    size: Optional[int] = None
    chunk_ids: List[str] = field(default_factory=list)

# --- Manifest Helper Functions ---
def load_manifest() -> Dict[str, dict]:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}

def save_manifest(manifest: Dict[str, dict]):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)

# --- Backend Functions ---
def build_or_update_faiss_index(document: Document, file_path: str):
    """Indexes a single uploaded document."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    manifest = load_manifest()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = None

    # Load and Chunk
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()
    pages = text_splitter.split_documents(raw_documents) 
    
    page_records = []
    for page in pages:
        page.metadata["read_access"] = list(sorted(document.read_access))
        page.metadata["name"] = document.name
        page.metadata["file_id"] = document.file_id
        page_records.append(page)

    # Add to FAISS
    if page_records:
        if vectorstore is None:
            vectorstore = FAISS.from_documents(page_records, embeddings)
            new_chunk_ids = [str(uuid.uuid4()) for _ in page_records]
            vectorstore = FAISS.from_documents(page_records, embeddings, ids=new_chunk_ids)
        else:
            new_chunk_ids = vectorstore.add_documents(page_records)
        
        document.chunk_ids = new_chunk_ids
        manifest[document.file_id] = asdict(document)
        
        vectorstore.save_local(FAISS_INDEX_PATH)
        save_manifest(manifest)
        return len(page_records)
    return 0

def delete_document_from_index(file_id: str):
    manifest = load_manifest()
    if file_id not in manifest:
        return False

    chunk_ids_to_delete = manifest[file_id].get("chunk_ids", [])
    
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        
        if chunk_ids_to_delete:
            vectorstore.delete(chunk_ids_to_delete)
            vectorstore.save_local(FAISS_INDEX_PATH)
        
        del manifest[file_id]
        save_manifest(manifest)
        
        # Optional: delete the physical file too
        file_path = os.path.join(UPLOAD_DIR, manifest.get(file_id, {}).get('name', ''))
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return True
    return False

def query_rag_agent(question: str, user_groups: List[str]):
    if not os.path.exists(FAISS_INDEX_PATH):
        return "Error: Database is empty. Please upload and index a document first.", []

    llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    def rbac_filter(metadata: dict) -> bool:
        doc_groups = metadata.get("read_access", [])
        return bool(set(user_groups) & set(doc_groups))

    retriever = vectorstore.as_retriever(search_kwargs={"filter": rbac_filter, "k": 4})

    system_prompt = (
        "You are a helpful AI assistant. Use the following pieces of retrieved context "
        "to answer the user's question. If the answer is not in the context, just say "
        "that you don't know based on the provided documents. Keep the answer clear and concise.\n\n"
        "<context>\n{context}\n</context>"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": question})
    return response["answer"], response.get("context", [])

# --- Streamlit UI ---
st.set_page_config(page_title="Secure RAG Chat", page_icon="🔒", layout="wide")
st.title("🔒 Secure RAG Chat with RBAC")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. API Key Input (Secure way to handle keys in Streamlit)
    groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key here. It will not be saved.")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        
    # 2. User Roles Input
    st.divider()
    st.header("👤 Your Access Profile")
    user_roles_input = st.text_input("Enter your roles (comma-separated)", value="guest")
    current_roles = [role.strip() for role in user_roles_input.split(",") if role.strip()]
    st.caption(f"Active Roles: `{current_roles}`")

    # 3. Document Uploader
    st.divider()
    st.header("📄 Upload & Index Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    doc_roles_input = st.text_input("Allowed Roles for this document (comma-separated)", value="all_users")
    
    if st.button("Index Document"):
        if not uploaded_file:
            st.warning("Please select a file to upload.")
        else:
            with st.spinner("Processing and Indexing..."):
                # Save file locally
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Create Document Object
                doc_roles = [role.strip() for role in doc_roles_input.split(",") if role.strip()]
                new_doc = Document(
                    file_id=str(uuid.uuid4())[:8], # Generate a short random ID
                    name=uploaded_file.name,
                    read_access=doc_roles
                )
                
                # Run Indexing
                chunks_added = build_or_update_faiss_index(new_doc, file_path)
                if chunks_added > 0:
                    st.success(f"Successfully indexed {chunks_added} chunks for '{uploaded_file.name}'.")
                else:
                    st.error("Indexing failed or no text found.")

    # 4. Database Manager
    st.divider()
    st.header("🗄️ Manage Database")
    manifest = load_manifest()
    if manifest:
        for doc_id, doc_data in manifest.items():
            st.write(f"**{doc_data['name']}**")
            st.caption(f"Roles: {doc_data['read_access']}")
            if st.button(f"Delete", key=f"del_{doc_id}"):
                if delete_document_from_index(doc_id):
                    st.success("Deleted! Refreshing...")
                    st.rerun()
    else:
        st.info("Database is empty.")


# --- Main Chat Interface ---
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("View Retrieved Context"):
                for i, chunk in enumerate(message["context"]):
                    st.markdown(f"**Chunk {i+1} (Source: {chunk.metadata.get('name', 'Unknown')}):**")
                    st.write(chunk.page_content)

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter your Groq API Key in the sidebar first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                answer, retrieved_docs = query_rag_agent(prompt, current_roles)
                
                st.markdown(answer)
                
                # Show context in an expander
                if retrieved_docs:
                    with st.expander(f"View Retrieved Context ({len(retrieved_docs)} chunks used)"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Chunk {i+1} (Source: {doc.metadata.get('name', 'Unknown')}) | Roles allowed: {doc.metadata.get('read_access')}**")
                            st.write(doc.page_content)
                
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "context": retrieved_docs
        })
