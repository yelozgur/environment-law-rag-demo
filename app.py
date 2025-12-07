import os
import faiss
import numpy as np
import fitz
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/intfloat/multilingual-e5-small"
HF_TOKEN = os.getenv("HF_TOKEN")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class RAG:
    def __init__(self, pdf_path, index_path="index.faiss", store_path="chunks.npy"):
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.store_path = store_path

        if not os.path.exists(index_path):
            print("🔧 FAISS index bulunamadı. Yeni index oluşturuluyor...")
            self.chunks = self._extract_text(pdf_path)
            self.embeddings = self._embed_batch(self.chunks)
            self.index = self._build_faiss(self.embeddings)

            faiss.write_index(self.index, index_path)
            np.save(store_path, self.chunks)
        else:
            print("📦 FAISS index bulundu. Yükleniyor...")
            self.index = faiss.read_index(index_path)
            self.chunks = np.load(store_path, allow_pickle=True)

    def _extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        chunks = []

        for page in doc:
            text = page.get_text().strip()
            if text:
                chunks.append(text)

        print(f"📄 Toplam {len(chunks)} doküman parçası çıkarıldı.")
        return np.array(chunks, dtype=object)

    def _embed_text(self, text):
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        data = {"inputs": text}

        response = requests.post(HF_API_URL, headers=headers, json=data)
        response.raise_for_status()

        return np.array(response.json(), dtype=np.float32)

    def _embed_batch(self, texts):
        return np.vstack([self._embed_text(t) for t in texts])

    def _build_faiss(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def ask_lawyer(self, query):
        q_emb = self._embed_text(query)
        scores, idx = self.index.search(q_emb.reshape(1, -1), 5)

        retrieved = "\n\n".join(self.chunks[i] for i in idx[0])

        prompt = f"""
Sen bir çevre hukuku uzmanı avukatsın.

Kurallar:
- Belgede yazmayan bir bilgi uydurma.
- Maddelere referans ver (varsa).
- Metne dayanarak açıklama yap.
- Anlaşılır, profesyonel hukuki Türkçe kullan.

### SORU:
{query}

### İLGİLİ MEVZUAT PARÇASI:
{retrieved}

### YANIT:
"""

        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return completion.choices[0].message.content
