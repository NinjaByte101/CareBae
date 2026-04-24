import os
import json
import time
import uuid
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
from typing import List, Dict, Any, Tuple

# ---- Embeddings & PDF processing ----
# Lightweight local embedding to keep deploy simple; you can swap to HuggingFace if needed.
import numpy as np
from pathlib import Path
from PyPDF2 import PdfReader

# ---- Groq LLM ----
# pip install groq
from groq import Groq

# -----------------------------
# Firebase Init
# -----------------------------
if not firebase_admin._apps:
    cred_dict = json.loads(st.secrets["FIREBASE_CREDENTIALS_JSON"])
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -----------------------------
# Config
# -----------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
PDF_PATH = st.secrets.get("PDF_PATH", "data/pdfs")
MODEL_NAME = "llama-3.1-8b-instant"  # fast, good balance; you can switch to 70B if needed

# -----------------------------
# Simple embedding function
# -----------------------------
def simple_embed(text: str) -> List[float]:
    # Deterministic pseudo-embedding: hash-based numeric features + length stats
    # Replace with sentence-transformers for production-grade quality.
    h = abs(hash(text)) % (10**8)
    vec = np.array([
        len(text),
        sum(c.isalpha() for c in text),
        sum(c.isdigit() for c in text),
        sum(c.isspace() for c in text),
        h % 997, h % 991, h % 983, h % 977
    ], dtype=np.float32)
    # Normalize
    norm = np.linalg.norm(vec) + 1e-9
    return (vec / norm).tolist()

def cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

# -----------------------------
# PDF ingestion & chunking
# -----------------------------
def read_pdf_text(pdf_file: Path) -> str:
    reader = PdfReader(str(pdf_file))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(tokens):
            break
    return chunks

def ingest_pdfs_to_firestore(pdf_dir: str, user_id: str = None, global_store: bool = True) -> int:
    """
    Ingest PDFs from pdf_dir, chunk, embed, and store in Firestore.
    If user_id is provided, store under user_embeddings/{uid}/vectors.
    If global_store is True, also store under pdf_embeddings/global_vectors.
    Returns number of chunks ingested.
    """
    base = Path(pdf_dir)
    if not base.exists():
        return 0

    count = 0
    for pdf in base.glob("*.pdf"):
        text = read_pdf_text(pdf)
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            emb = simple_embed(ch)
            chunk_id = f"{pdf.stem}-{i}-{uuid.uuid4().hex[:8]}"
            doc_data = {
                "text": ch,
                "embedding": emb,
                "source": pdf.name,
                "created_at": firestore.SERVER_TIMESTAMP
            }
            if user_id:
                db.collection("user_embeddings").document(user_id)\
                  .collection("vectors").document(chunk_id).set(doc_data)
            if global_store:
                db.collection("pdf_embeddings").document("global_vectors")\
                  .collection("vectors").document(chunk_id).set(doc_data)
            count += 1
    return count

# -----------------------------
# Retrieval from Firestore
# -----------------------------
def fetch_embeddings(user_id: str, use_global: bool = True) -> List[Dict[str, Any]]:
    vecs = []
    # user-specific
    user_ref = db.collection("user_embeddings").document(user_id).collection("vectors")
    for doc in user_ref.stream():
        d = doc.to_dict()
        if "embedding" in d and "text" in d:
            vecs.append(d)
    # global PDFs
    if use_global:
        glob_ref = db.collection("pdf_embeddings").document("global_vectors").collection("vectors")
        for doc in glob_ref.stream():
            d = doc.to_dict()
            if "embedding" in d and "text" in d:
                vecs.append(d)
    return vecs

def top_k_context(query: str, user_id: str, k: int = 5) -> List[str]:
    q_emb = simple_embed(query)
    candidates = fetch_embeddings(user_id=user_id, use_global=True)
    scored = []
    for c in candidates:
        sim = cosine_sim(q_emb, c["embedding"])
        scored.append((sim, c["text"], c.get("source", "unknown")))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f"[{src}] {txt}" for _, txt, src in scored[:k]]

# -----------------------------
# Memory summary
# -----------------------------
def get_memory_summary(user_id: str) -> str:
    doc = db.collection("users").document(user_id).get()
    if doc.exists:
        data = doc.to_dict()
        return data.get("memory_summary", "")
    return ""

def update_memory_summary(user_id: str, new_user_q: str, new_assistant_a: str, max_len: int = 1200):
    prev = get_memory_summary(user_id)
    # Simple rolling summary—replace with LLM summarization if desired
    combined = (prev + "\n- Q: " + new_user_q + "\n  A: " + new_assistant_a).strip()
    # Truncate to keep it lightweight
    if len(combined) > max_len:
        combined = combined[-max_len:]
    db.collection("users").document(user_id).update({"memory_summary": combined})

# -----------------------------
# Store messages
# -----------------------------
def store_message(user_id: str, role: str, content: str):
    db.collection("user_messages").document(user_id).collection("messages").add({
        "role": role,
        "content": content,
        "created_at": firestore.SERVER_TIMESTAMP
    })

def load_recent_messages(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    msgs_ref = db.collection("user_messages").document(user_id).collection("messages")\
                 .order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
    msgs = []
    for doc in msgs_ref.stream():
        msgs.append(doc.to_dict())
    return list(reversed(msgs))

# -----------------------------
# Groq LLM call
# -----------------------------
def call_groq(system_prompt: str, user_prompt: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    chat = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return chat.choices[0].message.content.strip()

# -----------------------------
# Session State Defaults
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# HEADER
# -----------------------------
st.title("🌸 CareBae")
st.subheader("Your Safe Period Health Companion")
st.info(
    "Educational use only. CareBae does not provide medical diagnosis or prescriptions. "
    "For severe or unusual symptoms, please consult a qualified doctor."
)

# -----------------------------
# AUTH / LOGOUT
# -----------------------------
if not st.session_state.logged_in:
    tab_login, tab_signup = st.tabs(["Login", "Create Account"])

    # ---------- LOGIN ----------
    with tab_login:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_btn"):
            try:
                user = auth.get_user_by_email(login_email)
                user_doc_ref = db.collection("users").document(user.uid)
                user_doc = user_doc_ref.get()

                if user_doc.exists:
                    st.session_state.logged_in = True
                    st.session_state.user = {"uid": user.uid, **user_doc.to_dict()}
                    st.session_state.messages = []
                    # Optional: ingest PDFs for this user on first login
                    ingest_pdfs_to_firestore(PDF_PATH, user_id=user.uid, global_store=True)
                    st.rerun()
                else:
                    st.error("User profile not found")
            except Exception as e:
                st.error(f"Login failed: {e}")

    # ---------- SIGNUP ----------
    with tab_signup:
        su_username = st.text_input("Username", key="su_username")
        su_email = st.text_input("Email", key="su_email")
        su_password = st.text_input("Password", type="password", key="su_password")

        if st.button("Create Account", key="signup_btn"):
            try:
                user = auth.create_user(email=su_email, password=su_password)
                db.collection("users").document(user.uid).set({
                    "username": su_username,
                    "email": su_email,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "memory_summary": ""
                })
                st.session_state.logged_in = True
                st.session_state.user = {"uid": user.uid, "username": su_username, "email": su_email}
                st.session_state.messages = []
                # Optional: ingest PDFs for this user on signup
                ingest_pdfs_to_firestore(PDF_PATH, user_id=user.uid, global_store=True)
                st.rerun()
            except Exception as e:
                st.error(f"Signup failed: {e}")

# -----------------------------
# BLOCK CHAT IF NOT LOGGED IN
# -----------------------------
if not st.session_state.logged_in:
    st.stop()

uid = st.session_state.user["uid"]

# -----------------------------
# WELCOME MESSAGE
# -----------------------------
WELCOME_MESSAGE = (
    f"🌸**Hi {st.session_state.user.get('username','there')}!**\n\n"
    "I'm CareBae!\n\n"
    "I'm here to help you understand **periods and menstrual health** "
    "in a safe, private, and judgment-free way.\n\n"
    "**You can ask me about:**\n"
    "• Menstrual cycles\n"
    "• Period pain & symptoms\n"
    "• Hygiene & general care\n"
)

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": WELCOME_MESSAGE})

def is_greeting(text):
    return text.lower().strip() in [
        "hi", "hello", "hey", "hii", "hiii",
        "good morning", "good afternoon", "good evening"
    ]

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("🌸 About CareBae")
    st.write("CareBae is an AI-powered chatbot designed to spread awareness about menstrual and period health.")
    st.divider()

    if st.session_state.logged_in:
        if st.button("Logout", key="sidebar_logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.subheader("Login")
        login_email = st.text_input("Email", key="sidebar_login_email")
        login_password = st.text_input("Password", type="password", key="sidebar_login_password")
        if st.button("Login", key="sidebar_login_btn"):
            try:
                user = auth.get_user_by_email(login_email)
                user_doc = db.collection("users").document(user.uid).get()
                if user_doc.exists:
                    st.session_state.logged_in = True
                    st.session_state.user = {"uid": user.uid, **user_doc.to_dict()}
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error("User profile not found")
            except Exception as e:
                st.error(f"Login failed: {e}")

        st.divider()
        st.subheader("Create Account")
        su_username = st.text_input("Username", key="sidebar_su_username")
        su_email = st.text_input("Email", key="sidebar_su_email")
        su_password = st.text_input("Password", type="password", key="sidebar_su_password")
        if st.button("Create Account", key="sidebar_signup_btn"):
            try:
                user = auth.create_user(email=su_email, password=su_password)
                db.collection("users").document(user.uid).set({
                    "username": su_username,
                    "email": su_email,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "memory_summary": ""
                })
                st.session_state.logged_in = True
                st.session_state.user = {"uid": user.uid, "username": su_username, "email": su_email}
                st.session_state.messages = []
                st.rerun()
            except Exception as e:
                st.error(f"Signup failed: {e}")

    st.divider()
    st.subheader("PDF ingestion")
    st.write(f"Backend folder: `{PDF_PATH}`")
    if st.button("Ingest PDFs now"):
        n = ingest_pdfs_to_firestore(PDF_PATH, user_id=uid, global_store=True)
        st.success(f"Ingested {n} chunks from PDFs.")

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# CHAT INPUT
# -----------------------------
user_input = st.chat_input("Ask anything about periods, symptoms, or hygiene...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    store_message(uid, "user", user_input)

    if is_greeting(user_input):
        reply = f"🌸I'm **CareBae**. How can I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": reply})
        store_message(uid, "assistant", reply)
        st.rerun()
    else:
        # Build retrieval-augmented prompt
        contexts = top_k_context(user_input, user_id=uid, k=6)
        memory_summary = get_memory_summary(uid)
        recent_msgs = load_recent_messages(uid, limit=10)

        system_prompt = (
            "You are CareBae, a safe, educational menstrual health companion. "
            "Provide general information only—do not diagnose, prescribe, or give medication advice. "
            "Encourage consulting qualified doctors for severe or unusual symptoms. "
            "Use the provided context and memory to answer clearly and compassionately."
        )

        user_prompt = (
            f"User question:\n{user_input}\n\n"
            f"Relevant PDF context (top matches):\n" +
            "\n\n".join([f"- {c}" for c in contexts]) +
            "\n\n" +
            f"User memory summary:\n{memory_summary or '(none)'}\n\n" +
            "Recent messages:\n" +
            "\n".join([f"- {m['role']}: {m['content']}" for m in recent_msgs]) +
            "\n\n" +
            "Answer with clear, empathetic guidance. If the question suggests severe pain, heavy bleeding, or missed periods, "
            "include a gentle reminder to consult a qualified doctor."
        )

        try:
            llm_answer = call_groq(system_prompt, user_prompt)
        except Exception as e:
            llm_answer = (
                "I couldn't generate a response right now. Please try again in a moment. "
                "If you experience severe pain, heavy bleeding, or missed periods, consult a qualified doctor."
            )

        # Append assistant message
        st.session_state.messages.append({"role": "assistant", "content": llm_answer})
        store_message(uid, "assistant", llm_answer)

        # Update memory summary
        update_memory_summary(uid, user_input, llm_answer)

        st.rerun()