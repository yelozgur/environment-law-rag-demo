import os
import streamlit as st
import fitz
import numpy as np
import json
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
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
def init_embedding_model():
    """Embedding modelini yükle"""
    try:
        # Hafif, hızlı bir model
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except Exception as e:
        st.error(f"Embedding model yüklenemedi: {e}")
        return None

@st.cache_resource  
def init_chroma_client():
    """ChromaDB client'ını başlat"""
    try:
        # Streamlit Cloud için persist directory
        persist_dir = "./chroma_db"
        Path(persist_dir).mkdir(exist_ok=True)
        
        client = chromadb.PersistentClient(path=persist_dir)
        return client
    except Exception as e:
        st.error(f"ChromaDB başlatma hatası: {e}")
        return None

class ChromaRAGSystem:
    def __init__(self):
        """ChromaDB tabanlı RAG sistemi başlat"""
        self.embedding_model = init_embedding_model()
        self.chroma_client = init_chroma_client()
        
        # Groq API
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama3-8b-8192"
        
        # Dosya yolları
        self.pdf_path = "documents/cevre_yasasi.pdf"
        self.metadata_path = "vectorstore/metadata.json"
        
        # Koleksiyon adı
        self.collection_name = "environment_law_docs"
        
        # Klasörleri oluştur
        Path("documents").mkdir(exist_ok=True)
        Path("vectorstore").mkdir(exist_ok=True)
        
        # Session state'i başlat
        if 'vectorstore_loaded' not in st.session_state:
            st.session_state.vectorstore_loaded = False
        if 'chunks_count' not in st.session_state:
            st.session_state.chunks_count = 0
            
        # Vector store'u yükle
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """ChromaDB vector store'u yükle"""
        try:
            # Koleksiyonları listele
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                self.collection = self.chroma_client.get_collection(self.collection_name)
                count = self.collection.count()
                
                st.session_state.vectorstore_loaded = True
                st.session_state.chunks_count = count
                
                # Metadata yükle
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                else:
                    self.metadata = {"source": self.pdf_path, "chunks_count": count}
                
                return True
            else:
                st.session_state.vectorstore_loaded = False
                return False
                
        except Exception as e:
            st.warning(f"Vector store yüklenemedi: {e}")
            st.session_state.vectorstore_loaded = False
            return False
    
    def extract_text_from_pdf(self, pdf_path=None):
        """PDF'den metin çıkar ve parçalara ayır"""
        chunks = []
        metadata_list = []
        
        try:
            # PDF yolu
            if pdf_path is None:
                pdf_path = self.pdf_path
            
            if not os.path.exists(pdf_path):
                st.error(f"PDF dosyası bulunamadı: {pdf_path}")
                return [], []
            
            # PDF'den metin çıkar
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                
                if text:
                    # Sayfayı paragraflara böl
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    
                    for para_num, paragraph in enumerate(paragraphs):
                        if 100 < len(paragraph) < 2000:  # Boyut kontrolü
                            chunks.append(paragraph)
                            metadata_list.append({
                                "page": page_num + 1,
                                "paragraph": para_num + 1,
                                "source": os.path.basename(pdf_path)
                            })
            
            doc.close()
            
            if chunks:
                st.success(f"✅ {len(chunks)} metin parçası çıkarıldı")
            else:
                st.warning("PDF'den metin çıkarılamadı!")
            
            return chunks, metadata_list
            
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
    
    def create_vectorstore(self):
        """PDF'den vector store oluştur"""
        if not os.path.exists(self.pdf_path):
            st.error(f"PDF dosyası bulunamadı: {self.pdf_path}")
            return False
        
        # Metin çıkar
        with st.spinner("📄 PDF işleniyor..."):
            chunks, metadata_list = self.extract_text_from_pdf()
        
        if not chunks:
            st.error("PDF'den metin çıkarılamadı!")
            return False
        
        # Embedding oluştur
        with st.spinner("🔨 Embedding'ler oluşturuluyor..."):
            embeddings = self.create_embeddings(chunks)
            
            if embeddings is None:
                return False
        
        # ChromaDB'ye ekle
        with st.spinner("🏗️ Vector store oluşturuluyor..."):
            try:
                # Eski koleksiyonu sil (varsa)
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                except:
                    pass
                
                # Yeni koleksiyon oluştur
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
                )
                
                # Batch ekleme
                batch_size = 50
                for i in range(0, len(chunks), batch_size):
                    end_idx = min(i + batch_size, len(chunks))
                    
                    batch_chunks = chunks[i:end_idx]
                    batch_embeddings = embeddings[i:end_idx]
                    batch_metadata = metadata_list[i:end_idx]
                    
                    # ID'ler oluştur
                    ids = [f"chunk_{j}" for j in range(i, end_idx)]
                    
                    # Koleksiyona ekle
                    self.collection.add(
                        embeddings=batch_embeddings.tolist(),
                        documents=batch_chunks,
                        metadatas=batch_metadata,
                        ids=ids
                    )
                
                # Metadata kaydet
                metadata = {
                    "source": self.pdf_path,
                    "chunks_count": len(chunks),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
                }
                
                with open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                # Session state'i güncelle
                st.session_state.vectorstore_loaded = True
                st.session_state.chunks_count = len(chunks)
                self.metadata = metadata
                
                st.success(f"✅ Vector store oluşturuldu: {len(chunks)} parça")
                return True
                
            except Exception as e:
                st.error(f"Vector store oluşturma hatası: {e}")
                return False
    
    def search(self, query, k=5):
        """Vector store'da benzer parçaları ara"""
        if not st.session_state.get('vectorstore_loaded', False):
            return []
        
        try:
            # ChromaDB'de ara
            results = self.collection.query(
                query_texts=[query],
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
                        'source': results['metadatas'][0][i].get('source', 'Unknown')
                    })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Arama hatası: {e}")
            return []
    
    def ask_question(self, query, k=5):
        """Soru sor ve yanıt al"""
        if not st.session_state.get('vectorstore_loaded', False):
            return {
                "answer": "Lütfen önce PDF yükleyin ve vector store oluşturun.",
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
    
    def clear_vectorstore(self):
        """Vector store'u temizle"""
        try:
            # Koleksiyonu sil
            self.chroma_client.delete_collection(self.collection_name)
            
            # Metadata dosyasını sil
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            
            # ChromaDB dizinini temizle
            import shutil
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                os.makedirs("./chroma_db")
            
            # Session state'i sıfırla
            st.session_state.vectorstore_loaded = False
            st.session_state.chunks_count = 0
            
            # Client'ı yeniden başlat
            self.chroma_client = init_chroma_client()
            
            return True
            
        except Exception as e:
            st.error(f"Temizleme hatası: {e}")
            return False

def main():
    """Ana Streamlit uygulaması"""
    st.title("⚖️ Çevre Hukuku Uzman Asistanı")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📂 Doküman Yönetimi")
        
        # PDF dosya durumu
        pdf_path = "documents/cevre_yasasi.pdf"
        pdf_exists = os.path.exists(pdf_path)
        
        if pdf_exists:
            file_size = os.path.getsize(pdf_path) / 1024 / 1024
            st.success(f"✅ PDF mevcut: {file_size:.2f} MB")
            
            if st.button("🔄 Vector Store Oluştur", type="primary", use_container_width=True):
                if 'rag_system' not in st.session_state:
                    st.session_state.rag_system = ChromaRAGSystem()
                
                rag = st.session_state.rag_system
                success = rag.create_vectorstore()
                if success:
                    st.success("✅ Vector store başarıyla oluşturuldu!")
                    time.sleep(2)
                    st.rerun()
        else:
            st.error("❌ PDF bulunamadı!")
            st.info("Lütfen `documents/cevre_yasasi.pdf` dosyasını yükleyin.")
        
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
        if st.session_state.get('vectorstore_loaded', False):
            st.success("✅ Vector store yüklü")
            chunks_count = st.session_state.get('chunks_count', 0)
            st.info(f"📊 {chunks_count} parça")
            
            # Metadata göster
            if os.path.exists("vectorstore/metadata.json"):
                try:
                    with open("vectorstore/metadata.json", 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    st.caption(f"Oluşturulma: {metadata.get('created_at', 'Unknown')}")
                except:
                    pass
        else:
            st.warning("⚠️ Vector store yüklenmedi")
        
        # Temizleme butonu
        st.markdown("---")
        if st.button("🗑️ Vector Store'u Temizle", type="secondary", use_container_width=True):
            if 'rag_system' in st.session_state:
                rag = st.session_state.rag_system
                success = rag.clear_vectorstore()
                if success:
                    st.success("✅ Vector store temizlendi!")
                    time.sleep(2)
                    st.rerun()
            else:
                st.warning("RAG sistemi başlatılmamış")
        
        # Yenile butonu
        st.markdown("---")
        if st.button("🔄 Sayfayı Yenile", type="secondary", use_container_width=True):
            st.rerun()
    
    # Ana içerik
    # RAG sistemini başlat
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = ChromaRAGSystem()
    
    rag = st.session_state.rag_system
    
    # Vector store durumu
    vectorstore_loaded = st.session_state.get('vectorstore_loaded', False)
    
    if vectorstore_loaded:
        # Başarılı yükleme
        chunks_count = st.session_state.get('chunks_count', 0)
        
        st.success(f"✅ Sistem hazır! {chunks_count} metin parçası yüklendi.")
        
        # Soru sorma bölümü
        st.subheader("❓ Soru Sor")
        
        query = st.text_area(
            "Çevre hukuku ile ilgili sorunuzu yazın:",
            placeholder="Örnek: Çevre kirliliği için cezai yaptırımlar nelerdir? Atık yönetimi yükümlülükleri nelerdir? Çevre izinleri nasıl alınır?",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🔍 Yanıt Al", type="primary", use_container_width=True) and query:
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
                        if result["sources"]:
                            first_page = result["sources"][0].get('page', 0)
                            if first_page > 0:
                                st.metric("İlk Sayfa", f"{first_page}")
                
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
            if st.button("📊 Durum", type="secondary", use_container_width=True):
                st.rerun()
    
    else:
        # Vector store yüklenemedi
        st.warning("""
        ### ⚠️ Vector Store Yüklenmedi
        
        **Adımlar:**
        1. **PDF kontrolü** → Sidebar'da PDF'nin mevcut olduğunu görün
        2. **Vector store oluştur** → "Vector Store Oluştur" butonuna tıklayın
        3. **Bekleyin** → PDF işlenecek ve embedding'ler oluşturulacak
        
        **📁 Mevcut Dosyalar:**
        ```
        /mount/src/environment-law-rag-demo/
        ├── documents/
        │   └── cevre_yasasi.pdf    ✅ VAR
        ├── vectorstore/
        │   ├── index.faiss         ⚠️ FAISS (kullanılmayacak)
        │   └── chunks.npy          ⚠️ FAISS (kullanılmayacak)
        ├── chroma_db/              ✅ ChromaDB için
        ├── app.py
        └── requirements.txt
        ```
        
        **ℹ️ Not:** Mevcut FAISS dosyaları kullanılmayacak, yeni ChromaDB vector store oluşturulacak.
        """)
        
        # Hızlı bilgiler
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("PDF Durumu", "✅ Mevcut" if pdf_exists else "❌ Eksik")
        
        with col2:
            st.metric("ChromaDB", "✅ Hazır")
        
        with col3:
            st.metric("Groq API", "✅ Hazır")
    
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
    # API key kontrolü
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not groq_key:
        st.error("""
        ### ⚠️ GROQ_API_KEY ayarlanmamış!
        
        **Çözüm:**
        1. **Streamlit Cloud Secrets**'ı kontrol edin
        2. **.env dosyası** oluşturun
        3. **Manuel olarak** API key girin
        
        **Secrets formatı (.streamlit/secrets.toml):**
        ```toml
        GROQ_API_KEY = "sk-..."
        ```
        """)
        
        # Debug için API key girişi
        with st.expander("🔑 API Key Girişi (Debug)"):
            groq_input = st.text_input("GROQ_API_KEY:", type="password")
            if groq_input:
                os.environ["GROQ_API_KEY"] = groq_input
                st.success("API Key kaydedildi! Sayfayı yenileyin.")
                if st.button("🔄 Yenile"):
                    st.rerun()
    else:
        main()
