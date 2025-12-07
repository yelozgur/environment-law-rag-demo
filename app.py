import os
import streamlit as st
import numpy as np
import fitz
import requests
import json
import tempfile
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Environment variables yükle
load_dotenv()

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="Çevre Hukuku Uzmanı",
    page_icon="⚖️",
    layout="wide"
)

# Cache'lenmiş fonksiyonlar
@st.cache_resource
def load_embedding_model():
    """Hafif embedding modelini yükle"""
    try:
        # CPU dostu, küçük model
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except Exception as e:
        st.error(f"Embedding model yüklenemedi: {e}")
        return None

@st.cache_resource
def init_chroma_client():
    """ChromaDB client'ını başlat"""
    try:
        # Streamlit Cloud için persist dizini
        persist_dir = "./chroma_db"
        Path(persist_dir).mkdir(exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                chroma_db_impl="duckdb+parquet",
                anonymized_telemetry=False
            )
        )
        return client
    except Exception as e:
        st.error(f"ChromaDB başlatma hatası: {e}")
        return None

class ChromaRAGSystem:
    def __init__(self):
        """ChromaDB tabanlı RAG sistemi"""
        self.embedding_model = load_embedding_model()
        self.chroma_client = init_chroma_client()
        
        # Groq API
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama3-8b-8192"
        
        # Koleksiyon adı
        self.collection_name = "cevre_hukuku"
        
        # Dosya yolları
        self.pdf_path = "documents/cevre_yasasi.pdf"
        self.metadata_path = "vectorstore/metadata.json"
        
        # Klasörleri oluştur
        self._ensure_directories()
        
        # Koleksiyonu yükle
        self._load_collection()
    
    def _ensure_directories(self):
        """Gerekli klasörleri oluştur"""
        Path("documents").mkdir(exist_ok=True)
        Path("vectorstore").mkdir(exist_ok=True)
    
    def _load_collection(self):
        """ChromaDB koleksiyonunu yükle veya oluştur"""
        try:
            if self.chroma_client is None:
                st.error("ChromaDB client başlatılamadı!")
                st.session_state.index_loaded = False
                return
            
            # Koleksiyonları listele
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                self.collection = self.chroma_client.get_collection(self.collection_name)
                count = self.collection.count()
                st.session_state.index_loaded = True
                st.session_state.chunks_count = count
                
                # Metadata yükle
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                else:
                    self.metadata = {"source": self.pdf_path, "chunks_count": count}
                
                return True
            else:
                st.session_state.index_loaded = False
                return False
                
        except Exception as e:
            st.warning(f"Koleksiyon yüklenemedi, yeni oluşturulacak: {e}")
            st.session_state.index_loaded = False
            return False
    
    def _create_collection(self):
        """Yeni koleksiyon oluştur"""
        try:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Çevre Hukuku Dokümanları"}
            )
            return True
        except Exception as e:
            st.error(f"Koleksiyon oluşturma hatası: {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_path=None, pdf_file=None):
        """PDF'den metin çıkar"""
        chunks = []
        page_chunk_map = []
        
        try:
            if pdf_file:
                # Geçici dosya oluştur
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_path = tmp_file.name
                doc_path = tmp_path
                is_temp = True
            else:
                doc_path = pdf_path or self.pdf_path
                is_temp = False
            
            if not os.path.exists(doc_path):
                st.error(f"PDF dosyası bulunamadı: {doc_path}")
                return []
            
            # PDF'den metin çıkar
            doc = fitz.open(doc_path)
            
            for page_num, page in enumerate(doc, 1):
                text = page.get_text().strip()
                if text:
                    # Paragraflara böl
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    for para in paragraphs:
                        if 50 < len(para) < 2000:  # Boyut kontrolü
                            chunks.append(para)
                            page_chunk_map.append(page_num)
            
            doc.close()
            
            # Geçici dosyayı temizle
            if is_temp:
                os.unlink(doc_path)
            
            if chunks:
                st.info(f"✅ {len(chunks)} metin parçası çıkarıldı")
            else:
                st.warning("PDF'den metin çıkarılamadı!")
            
            return chunks, page_chunk_map
            
        except Exception as e:
            st.error(f"PDF işleme hatası: {e}")
            return [], []
    
    def create_embeddings(self, texts):
        """Metinler için embedding oluştur"""
        if self.embedding_model is None:
            st.error("Embedding model yüklenemedi!")
            return None
        
        try:
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            st.error(f"Embedding oluşturma hatası: {e}")
            return None
    
    def create_index_from_existing_pdf(self):
        """Mevcut PDF'den index oluştur"""
        if not os.path.exists(self.pdf_path):
            st.error(f"PDF dosyası bulunamadı: {self.pdf_path}")
            return False
        
        with st.spinner("📄 Mevcut PDF işleniyor..."):
            chunks, page_chunk_map = self.extract_text_from_pdf(self.pdf_path)
        
        if not chunks:
            return False
        
        return self._add_to_collection(chunks, page_chunk_map, self.pdf_path)
    
    def create_index_from_new_pdf(self, pdf_file):
        """Yeni PDF'den index oluştur"""
        try:
            # PDF'i kaydet
            with open(self.pdf_path, 'wb') as f:
                f.write(pdf_file.getvalue())
            
            with st.spinner("📄 Yeni PDF işleniyor..."):
                chunks, page_chunk_map = self.extract_text_from_pdf(pdf_file=pdf_file)
            
            if not chunks:
                return False
            
            return self._add_to_collection(chunks, page_chunk_map, pdf_file.name)
            
        except Exception as e:
            st.error(f"PDF kaydetme hatası: {e}")
            return False
    
    def _add_to_collection(self, chunks, page_chunk_map, source_name):
        """Koleksiyona parçaları ekle"""
        try:
            # Embedding oluştur
            with st.spinner("🔨 Embedding'ler oluşturuluyor..."):
                embeddings = self.create_embeddings(chunks)
                
                if embeddings is None:
                    return False
            
            # Koleksiyon oluştur veya temizle
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass  # Koleksiyon yoksa sorun değil
            
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Batch halinde ekle
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                batch_chunks = chunks[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_pages = page_chunk_map[i:end_idx]
                
                # Metadata hazırla
                metadatas = [
                    {
                        "page": batch_pages[j],
                        "source": source_name,
                        "chunk_id": i + j
                    }
                    for j in range(len(batch_chunks))
                ]
                
                # ID'ler oluştur
                ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
                
                # Koleksiyona ekle
                self.collection.add(
                    embeddings=batch_embeddings.tolist(),
                    documents=batch_chunks,
                    metadatas=metadatas,
                    ids=ids
                )
            
            # Metadata kaydet
            metadata = {
                "source": source_name,
                "chunks_count": len(chunks),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "page_chunk_map": page_chunk_map
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Session state güncelle
            st.session_state.index_loaded = True
            st.session_state.chunks_count = len(chunks)
            self.metadata = metadata
            
            st.success(f"✅ Vector store oluşturuldu: {len(chunks)} parça")
            return True
            
        except Exception as e:
            st.error(f"Koleksiyona ekleme hatası: {e}")
            return False
    
    def search(self, query, k=5):
        """Benzer parçaları ara"""
        if not st.session_state.get('index_loaded', False):
            return []
        
        try:
            # Query embedding
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # ChromaDB'de ara
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Sonuçları formatla
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Cosine benzerliği
                    
                    formatted_results.append({
                        'text': doc,
                        'distance': float(distance),
                        'similarity': float(similarity),
                        'page': results['metadatas'][0][i].get('page', 0),
                        'chunk_id': results['metadatas'][0][i].get('chunk_id', 0),
                        'source': results['metadatas'][0][i].get('source', 'Unknown')
                    })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Arama hatası: {e}")
            return []
    
    def ask_question(self, query, k=5):
        """Soru sor ve yanıt al"""
        if not st.session_state.get('index_loaded', False):
            return {
                "answer": "Lütfen önce bir PDF yükleyin ve vector store oluşturun.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Benzer parçaları ara
        with st.spinner("🔍 İlgili dokümanlar aranıyor..."):
            results = self.search(query, k)
        
        if not results:
            return {
                "answer": "Üzgünüm, ilgili doküman bulunamadı.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Context oluştur
        context_parts = []
        for i, result in enumerate(results):
            page_info = f" [Sayfa {result['page']}]" if result.get('page') else ""
            context_parts.append(f"[Parça {i+1}{page_info}] {result['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Ortalama benzerlik
        avg_similarity = np.mean([r['similarity'] for r in results])
        
        # Prompt oluştur
        prompt = f"""Sen bir çevre hukuku uzmanı avukatsın.

Kullanıcı Sorusu: {query}

İlgili Mevzuat Parçaları:
{context}

Önemli Kurallar:
1. SADECE yukarıdaki mevzuat parçalarına dayan
2. Belgede olmayan bilgi EKLEME
3. Anlaşılır, profesyonel hukuki Türkçe kullan
4. Sayfa numaralarına referans ver
5. Eğer konu net değilse, "Belgede bu konu net belirtilmemiştir" de

Yanıt:"""
        
        # Groq API ile yanıt al
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "Sen bir çevre hukuku uzmanı avukatsın. Sadece verilen kaynaklara dayanarak cevap ver."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": results,
                "confidence": avg_similarity,
                "query": query
            }
            
        except Exception as e:
            st.error(f"API hatası: {e}")
            return {
                "answer": f"Üzgünüm, bir hata oluştu: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }

def main():
    """Ana Streamlit uygulaması"""
    st.title("⚖️ Çevre Hukuku Uzman Asistanı")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📂 Doküman Yönetimi")
        
        # PDF dosya durumu
        st.subheader("📄 Mevcut PDF")
        pdf_exists = os.path.exists("documents/cevre_yasasi.pdf")
        
        if pdf_exists:
            file_size = os.path.getsize("documents/cevre_yasasi.pdf") / 1024 / 1024
            st.success(f"✅ cevre_yasasi.pdf ({file_size:.2f} MB)")
            
            if st.button("🔄 Mevcut PDF'den Vector Store Oluştur", type="primary", use_container_width=True):
                if 'rag_system' not in st.session_state:
                    st.session_state.rag_system = ChromaRAGSystem()
                
                rag = st.session_state.rag_system
                success = rag.create_index_from_existing_pdf()
                if success:
                    st.success("✅ Vector store başarıyla oluşturuldu!")
                    time.sleep(2)
                    st.rerun()
        else:
            st.warning("⚠️ PDF bulunamadı")
        
        st.markdown("---")
        st.subheader("📤 Yeni PDF Yükle")
        
        uploaded_file = st.file_uploader(
            "Yeni PDF yükleyin",
            type=['pdf'],
            help="Mevcut PDF üzerine yazılacak"
        )
        
        if uploaded_file is not None:
            if st.button("📥 Yeni PDF ile Vector Store Oluştur", type="secondary", use_container_width=True):
                if 'rag_system' not in st.session_state:
                    st.session_state.rag_system = ChromaRAGSystem()
                
                rag = st.session_state.rag_system
                success = rag.create_index_from_new_pdf(uploaded_file)
                if success:
                    st.success("✅ Yeni PDF ile vector store oluşturuldu!")
                    time.sleep(2)
                    st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Ayarlar")
        
        k_results = st.slider(
            "Aranacak benzer doküman sayısı",
            min_value=1,
            max_value=10,
            value=5
        )
        
        st.markdown("---")
        st.subheader("🗄️ Vector Store Durumu")
        
        # Vector store durumu
        if 'rag_system' in st.session_state and st.session_state.get('index_loaded', False):
            st.success("✅ Vector store yüklü")
            chunks_count = st.session_state.get('chunks_count', 0)
            st.info(f"📊 {chunks_count} parça")
            
            # Metadata göster
            if os.path.exists("vectorstore/metadata.json"):
                try:
                    with open("vectorstore/metadata.json", 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    st.caption(f"Kaynak: {os.path.basename(metadata.get('source', 'Unknown'))}")
                    st.caption(f"Oluşturulma: {metadata.get('created_at', 'Unknown')}")
                except:
                    pass
        else:
            st.warning("⚠️ Vector store yüklenmedi")
        
        # Temizleme butonu
        st.markdown("---")
        if st.button("🗑️ Vector Store'u Temizle", type="secondary", use_container_width=True):
            try:
                # ChromaDB koleksiyonunu sil
                if 'rag_system' in st.session_state:
                    try:
                        st.session_state.rag_system.chroma_client.delete_collection(
                            st.session_state.rag_system.collection_name
                        )
                    except:
                        pass
                
                # Metadata dosyalarını sil
                for file in ["vectorstore/metadata.json"]:
                    if os.path.exists(file):
                        os.remove(file)
                
                # ChromaDB dizinini temizle
                import shutil
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")
                
                # Session state'i sıfırla
                if 'rag_system' in st.session_state:
                    del st.session_state.rag_system
                st.session_state.index_loaded = False
                
                st.success("✅ Vector store temizlendi!")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"Temizleme hatası: {e}")
    
    # Ana içerik alanı
    # RAG sistemini başlat
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = ChromaRAGSystem()
    
    rag = st.session_state.rag_system
    
    # Index durumunu kontrol et
    index_loaded = st.session_state.get('index_loaded', False)
    
    if not index_loaded:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.warning("""
            ### ⚠️ Vector Store Yüklenmedi
            
            **Ne yapabilirsiniz:**
            1. **Mevcut PDF'den vector store oluştur** → Sidebar'daki butonu kullanın
            2. **Yeni PDF yükle** → Sidebar'dan yeni PDF yükleyin
            
            **📁 Dosya Yapısı:**
            ```
            main/
            ├── documents/
            │   └── cevre_yasasi.pdf
            ├── vectorstore/
            │   └── metadata.json
            ├── chroma_db/ (otomatik oluşur)
            ├── app.py
            └── requirements.txt
            ```
            """)
        
        with col2:
            st.info("""
            **🎯 Özellikler:**
            - ✅ Python 3.13.9 uyumlu
            - ✅ FAISS gerekmez
            - ✅ ChromaDB kullanır
            - ✅ Local embedding
            - ✅ Persist storage
            """)
    
    # Soru sorma bölümü
    st.subheader("❓ Soru Sor")
    
    query = st.text_area(
        "Çevre hukuku ile ilgili sorunuzu yazın:",
        placeholder="Örnek: Çevre kirliliği için cezai yaptırımlar nelerdir? Atık yönetimi yükümlülükleri nelerdir?",
        height=100,
        disabled=not index_loaded
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔍 Yanıt Al", type="primary", disabled=not index_loaded, use_container_width=True) and query:
            if not index_loaded:
                st.error("Lütfen önce vector store oluşturun!")
                return
            
            # Yanıtı al
            result = rag.ask_question(query, k=k_results)
            
            # Yanıtı göster
            st.markdown("---")
            st.subheader("🤖 Uzman Yanıtı")
            
            with st.container():
                st.markdown(result["answer"])
                
                # İstatistikler
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Güven Skoru", f"{result['confidence']:.2%}")
                with cols[1]:
                    st.metric("Kullanılan Kaynak", len(result["sources"]))
                with cols[2]:
                    pages = [s.get('page', 0) for s in result['sources'] if s.get('page', 0) > 0]
                    if pages:
                        st.metric("Sayfa No", f"{pages[0]}")
            
            # Kaynakları göster
            if result["sources"]:
                with st.expander(f"📚 Kullanılan Kaynaklar ({len(result['sources'])})", expanded=False):
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"**Kaynak {i}**")
                        
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            st.metric("Benzerlik", f"{source['similarity']:.2%}")
                            if source.get('page'):
                                st.caption(f"Sayfa: {source['page']}")
                        
                        with col_b:
                            st.info(f"{source['text'][:500]}...")
                        
                        st.markdown("---")
    
    with col2:
        if st.button("🔄 Sayfayı Yenile", type="secondary", use_container_width=True):
            st.rerun()
    
    # Footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("⚡ Powered by Groq API")
    with col2:
        st.caption("🔍 ChromaDB Vector Search")
    with col3:
        st.caption("⚖️ Çevre Hukuku Uzmanı")

if __name__ == "__main__":
    # Environment variables kontrolü
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not groq_key:
        st.error("""
        ### ⚠️ GROQ_API_KEY ayarlanmamış!
        
        **Çözüm yolları:**
        
        1. **Streamlit Cloud Secrets:**
           ```toml
           # .streamlit/secrets.toml
           GROQ_API_KEY = "sk-..."
           ```
        
        2. **Local .env dosyası:**
           ```bash
           # .env dosyası oluşturun
           GROQ_API_KEY=sk-...
           ```
        """)
    else:
        main()
