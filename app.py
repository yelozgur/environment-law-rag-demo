import os
import streamlit as st
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from groq import Groq
import fitz  # PyMuPDF


# -------------------------------
# 1. CONFIG
# -------------------------------
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-70b-versatile"

DOC_PATH = "documents/cevre_yasasi.pdf"
CHUNKS_PATH = "vectorstore/chunks.npy"
INDEX_PATH = "vectorstore/index.faiss"


# -------------------------------
# 2. EMBEDDING MODEL
# -------------------------------
@st.cache_resource
def load_embedder():
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL)
    return tokenizer, model


tokenizer, embed_model = load_embedder()


def embed_text(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        output = embed_model(**tokens)
        embeddings = output.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().astype("float32")


# -------------------------------
# 3. PDF → Chunking
# -------------------------------
def load_pdf_chunks(pdf_path, chunk_size=700):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    # Split into chunks
    chunks = []
    words = full_text.split()
    current = []

    for w in words:
        current.append(w)
        if len(current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


# -------------------------------
# 4. FAISS Index Loader
# -------------------------------
def build_or_load_faiss():
    if os.path.exists(CHUNKS_PATH) and os.path.exists(INDEX_PATH):
        chunks = np.load(CHUNKS_PATH, allow_pickle=True)
        index = faiss.read_index(INDEX_PATH)
        return chunks, index

    st.warning("⚠️ FAISS index bulunamadı. Yeniden oluşturuluyor...")

    chunks = load_pdf_chunks(DOC_PATH)
    embeddings = []

    for c in st.progress_sequence(chunks, text="Embedding oluşturuluyor..."):
        embeddings.append(embed_text([c])[0])

    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    np.save(CHUNKS_PATH, np.array(chunks, dtype=object))
    faiss.write_index(index, INDEX_PATH)

    return chunks, index


chunks, index = build_or_load_faiss()


# -------------------------------
# 5. RETRIEVER
# -------------------------------
def retrieve(query, k=3):
    q_emb = embed_text([query])
    scores, idx = index.search(q_emb, k)
    return [chunks[i] for i in idx[0]]


# -------------------------------
# 6. GROQ LLM CALL
# -------------------------------
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])


def lawyer_prompt(query, context):
    return f"""
Sen bir çevre hukuku uzmanı avukatsın.

Kurallar:
- Belgede yer almayan hiçbir bilgiyi uydurma.
- Yorum yapman gerekirse “belgede yer alan bilgilere göre” diye başla.
- Mevzuat maddelerine referans ver (varsa).
- Açıklamayı anlaşılır ve profesyonel Türkçe hukuk diliyle yaz.
- Metindeki ifadeleri sadık kalarak kullan.

Soru:
{query}

İlgili mevzuat parçaları (bağlam):
{context}

Lütfen net ve maddeli şekilde açıkla.
"""


def ask_groq(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = lawyer_prompt(query, context)

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return completion.choices[0].message["content"]


# -------------------------------
# 7. STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Çevre Hukuku RAG Demo", layout="wide")

st.title("⚖️ Çevre Hukuku Danışma Hattı – RAG + Groq Demo")

st.write("""
Bu demo, Kıbrıs çevre mevzuatına ilişkin sorular için **LLM + RAG** yaklaşımı kullanır.  
Sorular PDF’teki gerçek metne göre yanıtlanır.
""")

query = st.text_input("Sorunuzu yazın:")

if st.button("Sorgula"):
    if not query.strip():
        st.error("Lütfen bir soru yazın.")
    else:
        with st.spinner("Yanıt hazırlanıyor..."):
            context_chunks = retrieve(query)
            answer = ask_groq(query, context_chunks)

            st.subheader("📌 Yanıt")
            st.write(answer)

            with st.expander("📄 Kullanılan Belgeler (RAG Çıktısı)"):
                for c in context_chunks:
                    st.write("— " + c[:500] + "...")
