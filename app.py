import os
import streamlit as st
import fitz
import numpy as np
from pathlib import Path
import json
import time
from groq import Groq
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Load environment
load_dotenv()

# Page config
st.set_page_config(page_title="Çevre Hukuku", layout="wide")

# Initialize
@st.cache_resource
def init_system():
    # Create directories
    Path("documents").mkdir(exist_ok=True)
    Path("vectorstore").mkdir(exist_ok=True)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get or create collection
    try:
        collection = client.get_or_create_collection(
            name="cevre_hukuku",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        )
        return client, collection, True
    except:
        return client, None, False

def main():
    st.title("⚖️ Çevre Hukuku")
    
    # Initialize
    client, collection, has_collection = init_system()
    
    # Sidebar
    with st.sidebar:
        st.header("📂 PDF İşleme")
        
        # Check PDF
        pdf_path = "documents/cevre_yasasi.pdf"
        has_pdf = os.path.exists(pdf_path)
        
        if has_pdf:
            size = os.path.getsize(pdf_path) / 1024 / 1024
            st.success(f"PDF: {size:.1f}MB")
            
            if st.button("🔄 Vector Store Oluştur"):
                with st.spinner("PDF işleniyor..."):
                    try:
                        # Extract text
                        doc = fitz.open(pdf_path)
                        texts = []
                        metadatas = []
                        
                        for page_num in range(len(doc)):
                            page = doc.load_page(page_num)
                            text = page.get_text().strip()
                            if text:
                                texts.append(text)
                                metadatas.append({"page": page_num + 1})
                        
                        doc.close()
                        
                        if texts:
                            # Create collection
                            collection = client.create_collection(
                                name="cevre_hukuku",
                                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                                )
                            )
                            
                            # Add to collection
                            ids = [f"page_{i+1}" for i in range(len(texts))]
                            collection.add(
                                documents=texts,
                                metadatas=metadatas,
                                ids=ids
                            )
                            
                            # Save metadata
                            metadata = {
                                "source": pdf_path,
                                "pages": len(texts),
                                "created": time.time()
                            }
                            with open("vectorstore/metadata.json", "w") as f:
                                json.dump(metadata, f)
                            
                            st.success(f"{len(texts)} sayfa eklendi!")
                            st.rerun()
                        
                    except Exception as e:
                        st.error(f"Hata: {e}")
        else:
            st.error("PDF bulunamadı!")
            
            # Upload PDF
            uploaded = st.file_uploader("PDF yükle", type=['pdf'])
            if uploaded:
                Path("documents").mkdir(exist_ok=True)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded.getvalue())
                st.success("PDF yüklendi! Sayfayı yenileyin.")
    
    # Main area
    if has_collection:
        st.subheader("❓ Soru Sor")
        
        query = st.text_input("Soru:")
        
        if query and st.button("Cevapla"):
            try:
                # Search
                results = collection.query(
                    query_texts=[query],
                    n_results=3
                )
                
                if results['documents']:
                    # Show context
                    with st.expander("📚 Bulunan Bilgiler"):
                        for i, doc in enumerate(results['documents'][0]):
                            st.write(f"**Sayfa {results['metadatas'][0][i]['page']}:**")
                            st.write(doc[:500] + "...")
                    
                    # Get answer (simplified)
                    st.subheader("🤖 Yanıt")
                    st.info("Bu bir demo. Gerçek yanıt için Groq API gerekli.")
                    
            except Exception as e:
                st.error(f"Hata: {e}")
    else:
        st.warning("Lütfen önce PDF yükleyin ve vector store oluşturun.")

if __name__ == "__main__":
    main()
