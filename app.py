import streamlit as st
import os
import numpy as np
import faiss
from groq import Groq
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch

# =========================
# Streamlit UI Setup
# =========================
st.set_page_config(page_title="Çevre Hukuku Danışma Asistanı", layout="wide")

st.title("⚖️ Çevre Hukuku Danışma Hattı – RAG + Groq Demo")
st.markdown("""
Bu asistan, **Kıbrıs çevre mevzuatına** ilişkin sorularınızı,
yüklenen **resmî PDF mevzuat dokümanlarından** yapay zekâ ile analiz ederek yanıtlar.

Model, **yalnızca belgedeki bilgilere dayanır**, hiçbir şekilde dış bilgi uydurmaz.  
""")

# =========================
# Embedding Model (TURKISH)
# =========================
@st.cache_resource
def load_embedder():
    model_name = "sabertazimi/turkish-stsb-xlm-r-multilingual"
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
# PDF Loading & Chunking
# =========================
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t + "\n"
    return text


def chunk_text(text, chunk_size=700):
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
# Build or Load FAISS
# =========================
def build_or_load_index():
    index_path = "vectorstore/index.faiss"
    chunks_path = "vectorstore/chunks.npy"

    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        chunks = np.load(chunks_path, allow_pickle=True).tolist()
        return index, chunks

    pdf_path = "documents/cevre_yasasi.pdf"
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)

    embeddings = embed_text(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    np.save(chunks_path, np.array(chunks, dtype=object))

    return index, chunks

index, chunks = build_or_load_index()


# =========================
# Search Function
# =========================
def search(query, index, chunks, k=4):
    q_emb = embed_text([query])
    scores, ids = index.search(q_emb, k)
    return [chunks[i] for i in ids[0]]


# =========================
# Groq Answer Generation
# =========================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def answer_with_groq(question, context):
    prompt = f"""
Sen bir çevre hukuku uzmanı avukatsın.

Kurallar:
- Belgede yazmayan bir bilgi uydurma.
- Her yanıtı **yalnızca verilen bağlamdaki** mevzuata dayandır.
- Varsa **madde numaralarıyla referans ver**.
- Metne dayanarak hukuki analiz yap.
- Açık, anlaşılır ve profesyonel Türkçe kullan.
- Bağlamda yoksa “Belgede bu konuda hüküm bulunmamaktadır” de.

--- BAĞLAM ---
{context}
--- SONU ---

SORU: {question}

Profesyonel hukuki yanıt:
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return response.choices[0].message.content


# =========================
# UI Input
# =========================
st.subheader("📨 Soru")
query = st.text_input("Çevre mevzuatına ilişkin sorunuzu yazın:")

if st.button("Yanıtla"):
    if not query.strip():
        st.error("Lütfen bir soru giriniz.")
    else:
        with st.spinner("Yanıt hazırlanıyor..."):
            retrieved = search(query, index, chunks)
            context = "\n\n".join(retrieved)
            answer = answer_with_groq(query, context)

        st.subheader("📌 Yanıt")
        st.write(answer)

        with st.expander("🔍 Kullanılan Mevzuat Metni"):
            st.write(context)

