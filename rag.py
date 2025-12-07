import os
import faiss
from pypdf import PdfReader
from google import generativeai as genai
import numpy as np
from tqdm import tqdm


EMBED_MODEL = "text-embedding-004"           # doğru yazım
LLM_MODEL = "gemini-1.5-flash-8b"            # v1beta with 0.8.5 → çalışır


class RAG:
    def __init__(self, pdf_path):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        self.embed_model = EMBED_MODEL
        self.llm = genai.GenerativeModel(model_name=LLM_MODEL)

        self.chunks = self._load_pdf_chunks(pdf_path)
        self.index = self._build_faiss_index(self.chunks)

    def _load_pdf_chunks(self, path):
        reader = PdfReader(path)
        text = "\n".join([page.extract_text() for page in reader.pages])
        chunks = text.split("\n\n")
        return chunks

    def _build_faiss_index(self, chunks):
        vectors = []
        for c in tqdm(chunks, desc="Embedding oluşturuluyor"):
            emb = genai.embed_content(
                model=self.embed_model,
                content=c
            )["embedding"]
            vectors.append(emb)

        vectors = np.array(vectors).astype("float32")
        dim = vectors.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        return index

    def search(self, query, top_k=3):
        q_emb = genai.embed_content(
            model=self.embed_model,
            content=query
        )["embedding"]

        q_emb = np.array(q_emb).astype("float32").reshape(1, -1)

        scores, idx = self.index.search(q_em_
