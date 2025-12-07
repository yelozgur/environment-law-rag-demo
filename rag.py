import faiss
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai
import streamlit as st

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=1500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
    start = end - overlap
    return chunks

def embed(text):
    res = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(res["embedding"], dtype="float32")

class RAG:
    def __init__(self, pdf_path):
        self.text = load_pdf(pdf_path)
        self.chunks = split_text(self.text)

        vectors = [embed(ch) for ch in self.chunks]
        self.vectors = np.vstack(vectors)

        self.index = faiss.IndexFlatL2(self.vectors.shape[1])
        self.index.add(self.vectors)

    def search(self, query, k=5):
        q_vec = embed(query).reshape(1, -1)
        _, idx = self.index.search(q_vec, k)
        return [self.chunks[i] for i in idx[0]]

    def ask_lawyer(self, query):
        context = "\n\n".join(self.search(query))

        system_prompt = """
Sen bir çevre hukuku uzmanı avukatsın.

Kurallar:
- Belgede yazmayan bir bilgi uydurma
- Maddelere referans ver (varsa)
- Metne dayanarak açıklama yap
- Anlaşılır, profesyonel hukuki Türkçe kullan
"""

        full_prompt = f"{system_prompt}\n\nSORU: {query}\n\nKAYNAK METİN:\n{context}"

        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(full_prompt)
        return response.text
