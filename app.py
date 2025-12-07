import streamlit as st
import os
import numpy as np
import faiss
from groq import Groq
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch


# =========================
# 1) Load embedding model
# =========================
@st.cache_resource
def load_embedder():
    model_name = "BAAI/bge-small-en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


tokenizer, model = load_embedder()


def embed_text(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        output = model(**tokens)
        embeddings = output.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().astype("float32")


# =========================
# 2) Load PDF & chunking
# =========================
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t + "\n"
    return text


def chunk_text(text, chunk_size=800):
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


# =========================
# 3) Build / load FAISS
# =========================
def build_or_load_index():
    index_path = "vectorstore/index.faiss"
    chunks_path = "vectorstore/chunks.npy"

    # if both exist → load
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        st.success("FAISS index bulundu — yükleniyor...")

        index = faiss.read_index(index_path)
        chunks = np.load(chunks_path, allow_pickle=True).tolist()
        return index, chunks

    # else → rebuild
    st.warning("⚠️ FAISS index bulunamadı. Yeniden oluşturuluyor...")

    pdf_path = "documents/cevre_yasasi.pdf"
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)

    embeddings = embed_text(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # save vectorstore
    faiss.write_index(index, index_path)
    np.save(chunks_path, np.array(chunks, dtype=object))

    st.success("✅ Yeni FAISS index oluşturuldu.")
    return index, chunks


index, chunks = build_or_load_index()


# =========================
# 4) Retrieval
# =========================
def search(query, index, chunks, k=3):
    q_emb = embed_text([query])
    scores, ids = index.search(q_emb, k)
    return [chunks[i] for i in ids[0]]


# =========================
# 5) Groq LLM answer
# =========================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


def answer_with_groq(question, context):
    prompt = f"""
Sen bir çevre hukuku asistanısın.
Sadece aşağıdaki bağlamdan alıntılar yaparak cevap ver:

--- BAĞLAM ---
{context}
--- SONU ---

SORU: {question}

Cevap:
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


# =========================
# 6) Streamlit UI
# =========================
st.title("⚖️ Çevre Hukuku Danışma – Groq RAG Demo")

query = st.text_input("Bir soru yazın (örn: 'ÇED nedir?')")

if st.button("Sorgula"):
    if not query.strip():
        st.error("Lütfen bir soru giriniz.")
    else:
        with st.spinner("Yanıt hazırlanıyor..."):
            retrieved = search(query, index, chunks)
            context = "\n\n".join(retrieved)
            answer = answer_with_groq(query, context)

        st.subheader("📌 Yanıt")
        st.write(answer)

        with st.expander("🔍 Kullanılan Bağlam"):
            st.write(context)
