import os
import json
import time
import uuid
import tempfile
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# ---- PDF processing ----
from pathlib import Path
from PyPDF2 import PdfReader
import PyPDF2

# ---- Embeddings & Vector DB ----
from sentence_transformers import SentenceTransformer
import pinecone
import numpy as np

# ---- Groq LLM ----
from groq import Groq

# -----------------------------
# Configuration from secrets
# -----------------------------
# Get all secrets
FIREBASE_CREDENTIALS_JSON = json.loads(st.secrets["FIREBASE_CREDENTIALS_JSON"])
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "carebae-docs")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
PDF_PATH = st.secrets.get("PDF_PATH", "data/pdfs")
MODEL_NAME = st.secrets.get("MODEL_NAME", "llama-3.1-8b-instant")

# -----------------------------
# Initialize Firebase
# -----------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_JSON)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -----------------------------
# Initialize Pinecone (v3+ syntax)
# -----------------------------
@st.cache_resource
def init_pinecone():
    """Initialize Pinecone client with v3+ syntax"""
    try:
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes] if hasattr(existing_indexes, '__iter__') else []
        
        if PINECONE_INDEX_NAME not in index_names:
            # Create index with appropriate dimension (using all-MiniLM-L6-v2 as default)
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,  # Dimension for all-MiniLM-L6-v2 model
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            st.sidebar.info(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
            # Wait for index to be ready
            time.sleep(30)
        
        # Connect to the index
        index = pc.Index(PINECONE_INDEX_NAME)
        return pc, index
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Pinecone: {e}")
        return None, None

pc, index = init_pinecone()

# -----------------------------
# Initialize Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for embeddings"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, good performance
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        # Fallback to simple embedding
        return None

embedding_model = load_embedding_model()

# -----------------------------
# Embedding Functions
# -----------------------------
def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using sentence transformer or fallback"""
    if embedding_model:
        return embedding_model.encode(text).tolist()
    else:
        # Fallback simple embedding
        return simple_embed_fallback(text)

def simple_embed_fallback(text: str) -> List[float]:
    """Simple deterministic embedding fallback"""
    import hashlib
    # Create a deterministic vector based on text hash
    hash_int = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    np.random.seed(hash_int)
    vec = np.random.randn(384).tolist()
    # Normalize
    norm = np.linalg.norm(vec)
    return [v / norm for v in vec] if norm > 0 else vec

# -----------------------------
# PDF Processing Functions
# -----------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """Split text into overlapping chunks with metadata"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "start_word": i,
            "end_word": min(i + chunk_size, len(words))
        })
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def process_pdf_file(pdf_path: Path, uploaded_by: str = "admin") -> List[Dict]:
    """Process a single PDF file into chunks with embeddings"""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        st.warning(f"No text extracted from {pdf_path.name}")
        return []
    
    chunks = chunk_text(text)
    
    if not chunks:
        st.warning(f"No chunks created from {pdf_path.name}")
        return []
    
    processed_chunks = []
    for idx, chunk_data in enumerate(chunks):
        chunk_id = f"{pdf_path.stem}_{idx}_{uuid.uuid4().hex[:8]}"
        embedding = get_embedding(chunk_data["text"])
        
        processed_chunks.append({
            "id": chunk_id,
            "text": chunk_data["text"],
            "embedding": embedding,
            "metadata": {
                "source": pdf_path.name,
                "uploaded_by": uploaded_by,
                "uploaded_at": datetime.utcnow().isoformat(),
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "file_size": os.path.getsize(pdf_path),
                "original_filename": pdf_path.name
            }
        })
    
    return processed_chunks

# -----------------------------
# Pinecone Operations (v3+ syntax)
# -----------------------------
def store_chunks_in_pinecone(chunks: List[Dict], namespace: str = "pdf_docs"):
    """Store chunks in Pinecone vector database"""
    if index is None:
        st.error("Pinecone index not initialized")
        return False
    
    try:
        if not chunks:
            st.warning("No chunks to store")
            return False
        
        # Prepare vectors for upsert using v3+ syntax
        vectors = []
        for chunk in chunks:
            vectors.append({
                "id": chunk["id"],
                "values": chunk["embedding"],
                "metadata": chunk["metadata"]
            })
        
        # Upsert in batches
        batch_size = 100
        successful_chunks = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            successful_chunks += len(batch)
        
        # Also store in Firestore for backup and text retrieval
        for chunk in chunks:
            doc_ref = db.collection("pdf_chunks").document(chunk["id"])
            doc_ref.set({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "namespace": namespace,
                "stored_at": firestore.SERVER_TIMESTAMP
            })
        
        # Log the upload in Firestore
        if chunks:
            first_chunk = chunks[0]
            upload_log_ref = db.collection("pdf_uploads").document()
            upload_log_ref.set({
                "filename": first_chunk["metadata"]["source"],
                "uploaded_by": first_chunk["metadata"]["uploaded_by"],
                "uploaded_at": firestore.SERVER_TIMESTAMP,
                "chunk_count": len(chunks),
                "namespace": namespace,
                "status": "success"
            })
        
        return True
    except Exception as e:
        st.error(f"Error storing chunks in Pinecone: {e}")
        
        # Log the error
        try:
            error_log_ref = db.collection("pdf_uploads").document()
            error_log_ref.set({
                "error": str(e),
                "uploaded_at": firestore.SERVER_TIMESTAMP,
                "status": "failed"
            })
        except:
            pass
        
        return False

def query_pinecone(query_text: str, namespace: str = "pdf_docs", top_k: int = 5) -> List[Dict]:
    """Query Pinecone for similar documents"""
    if index is None:
        return []
    
    try:
        query_embedding = get_embedding(query_text)
        
        # Query Pinecone using v3+ syntax
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        # Retrieve full text from Firestore
        retrieved_chunks = []
        for match in results.matches:
            chunk_id = match.id
            doc_ref = db.collection("pdf_chunks").document(chunk_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                retrieved_chunks.append({
                    "text": data["text"],
                    "metadata": match.metadata,
                    "score": match.score
                })
        
        return retrieved_chunks
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return []

def query_user_conversations(user_id: str, query_text: str, top_k: int = 3) -> List[Dict]:
    """Query user's past conversations for relevant context"""
    if index is None:
        return []
    
    try:
        query_embedding = get_embedding(query_text)
        
        # Query user-specific namespace
        namespace = f"user_{user_id}"
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        user_contexts = []
        for match in results.matches:
            if match.metadata:
                user_contexts.append({
                    "question": match.metadata.get("question", ""),
                    "answer": match.metadata.get("answer", ""),
                    "timestamp": match.metadata.get("timestamp", ""),
                    "score": match.score
                })
        
        return user_contexts
    except Exception as e:
        st.error(f"Error querying user conversations: {e}")
        return []

def store_user_conversation(user_id: str, question: str, answer: str):
    """Store user conversation in Pinecone for future reference"""
    if index is None:
        return
    
    try:
        # Create embedding from combined question + answer
        conversation_text = f"Q: {question}\nA: {answer}"
        embedding = get_embedding(conversation_text)
        
        conversation_id = f"conv_{uuid.uuid4().hex}"
        metadata = {
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in Pinecone using v3+ syntax
        index.upsert(
            vectors=[{
                "id": conversation_id,
                "values": embedding,
                "metadata": metadata
            }],
            namespace=f"user_{user_id}"
        )
        
        # Also store in Firestore for backup
        db.collection("user_conversations").document(user_id).collection("chats").document(conversation_id).set({
            "question": question,
            "answer": answer,
            "embedding": embedding,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        
    except Exception as e:
        st.error(f"Error storing user conversation: {e}")

# -----------------------------
# PDF Upload and Processing
# -----------------------------
def process_uploaded_pdfs(pdf_dir: str = None, uploaded_files: List = None, user_id: str = "admin"):
    """Process and store PDFs in vector database"""
    total_chunks = 0
    
    if pdf_dir and os.path.exists(pdf_dir):
        # Process existing PDFs in directory
        pdf_path = Path(pdf_dir)
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            with st.spinner(f"Processing {pdf_file.name}..."):
                chunks = process_pdf_file(pdf_file, uploaded_by=user_id)
                if chunks:
                    success = store_chunks_in_pinecone(chunks)
                    if success:
                        total_chunks += len(chunks)
                        st.success(f"✅ Processed {pdf_file.name}: {len(chunks)} chunks")
    
    if uploaded_files:
        # Process newly uploaded files
        for uploaded_file in uploaded_files:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    chunks = process_pdf_file(Path(temp_path), uploaded_by=user_id)
                    if chunks:
                        success = store_chunks_in_pinecone(chunks)
                        if success:
                            total_chunks += len(chunks)
                            st.success(f"✅ Processed {uploaded_file.name}: {len(chunks)} chunks")
                        else:
                            st.error(f"❌ Failed to store chunks for {uploaded_file.name}")
                    else:
                        st.warning(f"⚠️ No chunks created for {uploaded_file.name}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    return total_chunks

def get_uploaded_pdfs_stats():
    """Get statistics about uploaded PDFs"""
    try:
        # Get from Firestore
        uploads_ref = db.collection("pdf_uploads").order_by("uploaded_at", direction=firestore.Query.DESCENDING).limit(10)
        uploads = list(uploads_ref.stream())
        
        # Get from Pinecone if available
        total_vectors = 0
        if index is not None:
            try:
                stats = index.describe_index_stats()
                total_vectors = stats.total_vector_count
            except:
                total_vectors = 0
        
        return {
            "recent_uploads": [upload.to_dict() for upload in uploads],
            "total_vectors": total_vectors
        }
    except Exception as e:
        st.error(f"Error getting stats: {e}")
        return {"recent_uploads": [], "total_vectors": 0}

# -----------------------------
# Admin Functions
# -----------------------------
def display_admin_panel(user_id: str):
    """Display the admin panel for PDF management"""
    st.header("📁 Admin Panel - PDF Management")
    
    # Create tabs for different admin functions
    tab1, tab2, tab3 = st.tabs(["📤 Upload PDFs", "📊 View Statistics", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Upload PDF Documents")
        st.markdown("""
        Upload PDF documents to add them to the knowledge base. These documents will be:
        - Processed and split into chunks
        - Converted to vector embeddings
        - Stored in Pinecone for semantic search
        - Available to all users
        """)
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Select one or more PDF files to upload"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} PDF file(s)")
            
            # Show preview of selected files
            with st.expander("📋 Selected Files Preview"):
                for i, file in enumerate(uploaded_files):
                    st.write(f"{i+1}. **{file.name}** ({file.size / 1024:.1f} KB)")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size (words)", 300, 1000, 500, 50,
                                 help="Number of words per text chunk")
        with col2:
            overlap = st.slider("Chunk Overlap (words)", 50, 300, 100, 10,
                              help="Overlap between chunks for context preservation")
        
        # Process button
        if st.button("🚀 Process and Upload PDFs", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please select PDF files to upload first!")
            else:
                with st.spinner("Processing PDFs..."):
                    progress_bar = st.progress(0)
                    
                    # Process each file
                    total_processed = 0
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Update progress
                        progress = (i) / len(uploaded_files)
                        progress_bar.progress(progress)
                        
                        # Save to temp file and process
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            temp_path = tmp_file.name
                        
                        try:
                            # Extract text
                            text = extract_text_from_pdf(Path(temp_path))
                            if not text:
                                st.warning(f"No text extracted from {uploaded_file.name}")
                                continue
                            
                            # Chunk text
                            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                            
                            if chunks:
                                # Create embeddings and store
                                processed_chunks = []
                                for idx, chunk_data in enumerate(chunks):
                                    chunk_id = f"{uploaded_file.name.replace('.pdf', '')}_{idx}_{uuid.uuid4().hex[:8]}"
                                    embedding = get_embedding(chunk_data["text"])
                                    
                                    processed_chunks.append({
                                        "id": chunk_id,
                                        "text": chunk_data["text"],
                                        "embedding": embedding,
                                        "metadata": {
                                            "source": uploaded_file.name,
                                            "uploaded_by": user_id,
                                            "uploaded_at": datetime.utcnow().isoformat(),
                                            "chunk_index": idx,
                                            "total_chunks": len(chunks),
                                            "file_size": uploaded_file.size,
                                            "original_filename": uploaded_file.name
                                        }
                                    })
                                
                                # Store in Pinecone
                                if processed_chunks:
                                    success = store_chunks_in_pinecone(processed_chunks)
                                    if success:
                                        total_processed += len(processed_chunks)
                                        st.success(f"✅ {uploaded_file.name}: {len(processed_chunks)} chunks")
                                    else:
                                        st.error(f"❌ Failed to store {uploaded_file.name}")
                            else:
                                st.warning(f"No chunks created from {uploaded_file.name}")
                        
                        finally:
                            # Clean up
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                    
                    # Complete progress
                    progress_bar.progress(1.0)
                    
                    if total_processed > 0:
                        st.balloons()
                        st.success(f"🎉 Successfully processed {len(uploaded_files)} PDF(s) into {total_processed} total chunks!")
                        
                        # Clear uploaded files from session state
                        if 'uploaded_files' in st.session_state:
                            st.session_state.uploaded_files = []
                    else:
                        st.error("No chunks were processed. Please check your PDF files.")
    
    with tab2:
        st.subheader("Knowledge Base Statistics")
        
        # Get stats
        stats = get_uploaded_pdfs_stats()
        
        # Display stats in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Vectors", stats["total_vectors"])
        with col2:
            recent_count = len(stats["recent_uploads"])
            st.metric("Recent Uploads", recent_count)
        with col3:
            if stats["recent_uploads"]:
                latest = stats["recent_uploads"][0]
                st.metric("Latest Upload", latest.get("filename", "N/A"))
        
        # Recent uploads table
        st.subheader("Recent Uploads")
        if stats["recent_uploads"]:
            upload_data = []
            for upload in stats["recent_uploads"]:
                upload_time = upload.get("uploaded_at")
                if hasattr(upload_time, 'strftime'):
                    upload_time = upload_time.strftime("%Y-%m-%d %H:%M")
                elif isinstance(upload_time, str):
                    # Try to parse the string
                    try:
                        dt = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
                        upload_time = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                
                upload_data.append({
                    "Filename": upload.get("filename", "Unknown"),
                    "Chunks": upload.get("chunk_count", 0),
                    "Uploaded By": upload.get("uploaded_by", "Unknown"),
                    "Time": upload_time,
                    "Status": upload.get("status", "Unknown")
                })
            
            st.dataframe(upload_data, use_container_width=True)
        else:
            st.info("No uploads yet.")
        
        # Test search functionality
        st.subheader("Test Search")
        test_query = st.text_input("Enter a test query to search the knowledge base:")
        if test_query and st.button("Test Search"):
            with st.spinner("Searching..."):
                results = query_pinecone(test_query, top_k=3)
                if results:
                    st.success(f"Found {len(results)} relevant chunks:")
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} (Score: {result['score']:.3f}) - {result['metadata']['source']}"):
                            st.write(f"**Source:** {result['metadata']['source']}")
                            st.write(f"**Text:** {result['text'][:300]}...")
                else:
                    st.warning("No results found.")
    
    with tab3:
        st.subheader("Settings")
        
        # Pinecone settings
        st.write("**Pinecone Configuration**")
        st.code(f"""
        Index Name: {PINECONE_INDEX_NAME}
        API Key: {PINECONE_API_KEY[:10]}...{PINECONE_API_KEY[-10:] if len(PINECONE_API_KEY) > 20 else ''}
        Environment: {PINECONE_ENVIRONMENT}
        """)
        
        # Embedding model info
        st.write("**Embedding Model**")
        if embedding_model:
            st.success("✅ SentenceTransformer loaded (all-MiniLM-L6-v2)")
        else:
            st.error("❌ Embedding model not loaded")
        
        # System status
        st.write("**System Status**")
        if index is not None:
            try:
                index_stats = index.describe_index_stats()
                st.success("✅ Pinecone is connected and ready")
                st.write(f"Total vectors across all namespaces: {index_stats.total_vector_count}")
            except Exception as e:
                st.error(f"❌ Pinecone error: {e}")
        else:
            st.error("❌ Pinecone not initialized")
        
        # Reset button (for development)
        st.divider()
        st.write("**Danger Zone**")
        if st.button("Clear All Uploads (Development Only)", type="secondary"):
            st.warning("This will clear all uploaded data. Are you sure?")
            confirm = st.checkbox("I understand this will delete all data")
            if confirm and st.button("Confirm Clear All"):
                # Note: In production, you might want to implement proper deletion
                st.info("Data clearance would be implemented here")

# -----------------------------
# Groq LLM Integration
# -----------------------------
def call_groq_with_context(system_prompt: str, user_prompt: str, context: str = "") -> str:
    """Call Groq LLM with provided context"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if context:
            messages.append({
                "role": "user", 
                "content": f"Context information:\n{context}\n\nUser question: {user_prompt}"
            })
        else:
            messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again."

# -----------------------------
# Main Application
# -----------------------------
def main():
    st.set_page_config(
        page_title="CareBae - Women's Health Assistant",
        page_icon="🌸",
        layout="wide"
    )
    
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_admin" not in st.session_state:
        st.session_state.show_admin = False
    
    # Main header
    st.title("🌸 CareBae - Women's Health Assistant")
    st.markdown("""
    Your safe, private, and knowledgeable companion for menstrual health education.
    
    **⚠️ Important**: This is for educational purposes only. Always consult with 
    healthcare professionals for medical advice.
    """)
    
    # Sidebar for auth and navigation
    with st.sidebar:
        st.header("Account & Navigation")
        
        if not st.session_state.logged_in:
            # Login/Signup
            auth_tab = st.radio("Choose", ["Login", "Sign Up"], horizontal=True)
            
            if auth_tab == "Login":
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                
                if st.button("Login", use_container_width=True):
                    try:
                        user = auth.get_user_by_email(email)
                        user_doc = db.collection("users").document(user.uid).get()
                        
                        if user_doc.exists:
                            st.session_state.logged_in = True
                            st.session_state.user = {
                                "uid": user.uid,
                                "email": email,
                                **user_doc.to_dict()
                            }
                            st.session_state.messages = []
                            st.session_state.show_admin = False
                            st.success("Logged in successfully!")
                            st.rerun()
                        else:
                            st.error("User profile not found")
                    except Exception as e:
                        st.error(f"Login failed: {e}")
            
            else:  # Sign Up
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                
                if st.button("Create Account", use_container_width=True):
                    try:
                        user = auth.create_user(email=email, password=password)
                        db.collection("users").document(user.uid).set({
                            "username": username,
                            "email": email,
                            "created_at": firestore.SERVER_TIMESTAMP,
                            "is_admin": False  # Default to non-admin
                        })
                        
                        st.session_state.logged_in = True
                        st.session_state.user = {
                            "uid": user.uid,
                            "username": username,
                            "email": email,
                            "is_admin": False
                        }
                        st.session_state.messages = []
                        st.session_state.show_admin = False
                        st.success("Account created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")
        
        else:  # User is logged in
            st.success(f"Welcome, {st.session_state.user.get('username', 'User')}!")
            
            # Navigation
            st.divider()
            st.subheader("Navigation")
            
            # Main chat button
            if st.button("💬 Go to Chat", use_container_width=True):
                st.session_state.show_admin = False
                st.rerun()
            
            # Admin panel button (only for admins)
            if st.session_state.user.get("is_admin", False):
                if st.button("📁 Admin Panel", use_container_width=True):
                    st.session_state.show_admin = True
                    st.rerun()
            
            # Display user info
            st.divider()
            st.write("**Account Info**")
            st.write(f"Email: {st.session_state.user.get('email')}")
            st.write(f"Admin: {'✅ Yes' if st.session_state.user.get('is_admin') else '❌ No'}")
            
            # Logout button
            st.divider()
            if st.button("Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.session_state.messages = []
                st.session_state.show_admin = False
                st.rerun()
    
    # Main content area
    if not st.session_state.logged_in:
        # Show welcome screen when not logged in
        st.info("Please login or sign up to start chatting with CareBae!")
        
        # Display features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📚 Knowledge Base")
            st.write("Access information from verified health resources and uploaded documents")
        
        with col2:
            st.markdown("### 🔒 Privacy First")
            st.write("Your conversations are private and secure")
        
        with col3:
            st.markdown("### 🩺 Safe Information")
            st.write("Educational content only - we encourage doctor consultations for medical concerns")
    
    elif st.session_state.show_admin and st.session_state.user.get("is_admin", False):
        # Show admin panel
        display_admin_panel(st.session_state.user["uid"])
    
    else:
        # Show chat interface
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about women's health, periods, symptoms, or hygiene..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Store user message in Firestore
            db.collection("user_messages").document(st.session_state.user["uid"]).collection("messages").add({
                "role": "user",
                "content": prompt,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            
            # Generate response with context
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Step 1: Retrieve relevant PDF information
                    pdf_contexts = query_pinecone(prompt, top_k=3)
                    pdf_context_text = "\n\n".join([f"Source: {ctx['metadata']['source']}\n{ctx['text']}" 
                                                   for ctx in pdf_contexts]) if pdf_contexts else "No relevant PDF information found."
                    
                    # Step 2: Retrieve user's past conversations
                    user_contexts = query_user_conversations(st.session_state.user["uid"], prompt, top_k=2)
                    user_context_text = "\n\n".join([f"Previous Q: {ctx['question']}\nPrevious A: {ctx['answer']}" 
                                                    for ctx in user_contexts]) if user_contexts else "No previous conversation history."
                    
                    # Step 3: Construct system prompt with context
                    system_prompt = f"""You are CareBae, a compassionate and knowledgeable women's health assistant. 
                    
IMPORTANT GUIDELINES:
1. Provide educational information only - never diagnose, prescribe, or give medical advice
2. Always encourage consulting healthcare professionals for serious concerns
3. Be empathetic, non-judgmental, and supportive
4. Use clear, simple language that's easy to understand
5. If you don't know something, admit it and suggest consulting a doctor

KNOWLEDGE BASE CONTEXT (from uploaded PDFs):
{pdf_context_text}

USER'S PREVIOUS CONVERSATIONS (for context):
{user_context_text}

Based on the above information and your general knowledge, provide a helpful response."""
                    
                    # Step 4: Generate response
                    response = call_groq_with_context(system_prompt, prompt)
                    
                    # Step 5: Store conversation in Pinecone for future reference
                    store_user_conversation(st.session_state.user["uid"], prompt, response)
                    
                    # Step 6: Display response
                    st.markdown(response)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Store assistant message in Firestore
                    db.collection("user_messages").document(st.session_state.user["uid"]).collection("messages").add({
                        "role": "assistant",
                        "content": response,
                        "timestamp": firestore.SERVER_TIMESTAMP
                    })

# -----------------------------
# Make Admin Users
# -----------------------------
def make_user_admin(email: str):
    """Make a user an admin (run this once to set up admin users)"""
    try:
        user = auth.get_user_by_email(email)
        db.collection("users").document(user.uid).update({"is_admin": True})
        print(f"User {email} is now an admin")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
    
    
# make_user_admin("2305101270036@paruluniversity.ac.in")
make_user_admin("2305101010269@paruluniversity.ac.in")