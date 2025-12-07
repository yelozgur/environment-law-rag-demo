import streamlit as st
from groq import Groq
import faiss
import numpy as np
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch


# ---------------------------
# Load embedding model
# ---------------------------
@st.cache_resource
def load_embedder():
    model_name = "BAAI/bge-small-en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_embedder()


def embed_text(texts):
    """Embeds list of texts using bge-small-en"""
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**tokens)
        embeddings = output.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().astype("float32")


# ---------------------------
# Groq client
# ---------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ---------------------------
# PDF Functions
# ---------------------------
def load_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def split_text(text, chunk_size=800):
    words = text.split()
    chunks = []
    buf = []

    for w in words:
        buf.append(w)
        if len(buf) >= chunk_size:
            chunks.append(" ".join(buf))
            buf = []

    if buf:
        chunks.append(" ".join(buf))

    return chunks


def build_index(chunks):
    embeddings = embed_text(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index


def search(query, index, chunks, top_k=3):
    q = embed_text([query])
    distances, ids = index.search(q, top_k)
    return [chunks[i] for i in ids[0]]


def groq_answer(question, context):
    prompt = f"""
You are an assistant. Use ONLY the following context:

{context}

Question: {question}

Answer:
"""

    res = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return res.choices[0].message.content


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📘 Groq RAG (FAISS + Transformers) – GitHub Version")
st.write("Upload a PDF and query with Groq RAG")

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    text = load_pdf_text(pdf)
    chunks = split_text(text)
    index = build_index(chunks)

    st.session_state.chunks = chunks
    st.session_state.index = index

    st.success("PDF processed. Ask your question!")


question = st.text_input("Your question:")

if st.button("Answer"):
    if "index" not in st.session_state:
        st.error("Upload a PDF first.")
    else:
        retrieved = search(
            question,
            st.session_state.index,
            st.session_state.chunks
        )
        context = "\n\n".join(retrieved)
        answer = groq_answer(question, context)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            st.write(context)
