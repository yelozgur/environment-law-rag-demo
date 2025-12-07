import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai


class RAG:
    def __init__(self, api_key, documents):
        self.documents = documents

        # Embedding Model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode docs
        self.embeddings = self.model.encode(
            [d["text"] for d in documents], convert_to_numpy=True
        )

        # FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        # Google client
        self.client = genai.Client(api_key=api_key)

    def search(self, query, k=5):
        """Retrieve top-k docs"""
        q_em = self.model.encode([query], convert_to_numpy=True)

        # FIX: previous error — parenthesis closed and variable name corrected
        scores, idx = self.index.search(q_em, k)

        results = []
        for i in idx[0]:
            results.append(self.documents[i])

        return results

    def build_prompt(self, question, contexts):
        context_block = "\n\n".join(
            f"[{d['id']}] {d['text']}" for d in contexts
        )

        return f"""
You are an expert in Turkish environmental law.
Use ONLY the information in the CONTEXT. 
If information is missing, say "mevzuatta bunun karşılığı bulunmamaktadır".

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:
"""

    def ask(self, question):
        ctx = self.search(question)
        prompt = self.build_prompt(question, ctx)

        # FIX: correct model name
        response = self.client.models.generate_content(
            model="models/gemini-2.0-flash-exp",
            contents=prompt
        )

        return response.text, ctx
