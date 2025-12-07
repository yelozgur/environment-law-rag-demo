import faiss
import os
from pypdf import PdfReader
from google.generativeai import configure, GenerativeModel
import numpy as np
from tqdm import tqdm

configure(api_key=os.getenv("GEMINI_API_KEY"))


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


class RAG:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.index_path = "vectorstore/index.faiss"
        self.meta_path = "vectorstore/chunks.npy"

        os.makedirs("vectorstore", exist_ok=True)

        if not os.path.exists(self.index_path):
            print("⚠️ FAISS index bulunamadı. Yeniden oluşturuluyor...")
            self._build_index()
        else:
            print("ℹ️ FAISS index bulundu, yükleniyor...")

        self.index = faiss.read_index(self.index_path)
        self.chunks = np.load(self.meta_path, allow_pickle=True)
        self.model = GenerativeModel("gemini-1.5-flash")

    def _build_index(self):
        reader = PdfReader(self.pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        chunks = chunk_text(full_text)

        # Embeddings üret
        model = GenerativeModel("text-embedding-004")
        embeddings = []
        for ch in tqdm(chunks, desc="Embedding oluşturuluyor"):
            emb = model.embed_content(
                content=ch,
                model="text-embedding-004"
            )["embedding"]
            embeddings.append(emb)

        embeddings = np.array(embeddings).astype("float32")

        # FAISS index oluştur
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, self.index_path)
        np.save(self.meta_path, np.array(chunks, dtype=object))

        print("✔ FAISS index oluşturuldu!")

    def ask_lawyer(self, query):
        model = GenerativeModel("text-embedding-004")
        q_emb = model.embed_content(content=query)["embedding"]
        q_emb = np.array(q_emb).astype("float32").reshape(1, -1)

        D, I = self.index.search(q_emb, 3)
        retrieved = "\n".join(self.chunks[i] for i in I[0])

        prompt = f"""
Sen Kıbrıs'ın çevre mevzuatında uzman bir hukuk danışmanısın.
Aşağıdaki mevzuat parçalarına dayanarak profesyonel bir yanıt ver.

Mevzuat:
{retrieved}

Soru:
{query}

Cevap (hukuki, açıklayıcı):
"""

        lawyer_model = GenerativeModel("gemini-1.5-flash")
        response = lawyer_model.generate_content(prompt)
        return response.text
