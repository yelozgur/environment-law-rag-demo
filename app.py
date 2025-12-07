import os
import streamlit as st
import numpy as np
import fitz
import requests
from groq import Groq
from dotenv import load_dotenv
import faiss
import tempfile

# Environment variables yükle
load_dotenv()

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="Çevre Hukuku Uzmanı",
    page_icon="⚖️",
    layout="wide"
)

class StreamlitRAGSystem:
    def __init__(self):
        """RAG sistemini başlat"""
        # API ayarları
        self.HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/intfloat/multilingual-e5-small"
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.HF_HEADERS = {"Authorization": f"Bearer {self.HF_TOKEN}"}
        
        # Groq API
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama3-8b-8192"
        
        # FAISS index yolları
        self.index_path = "documents/index.faiss"
        self.chunks_path = "documents/chunks.npy"
        
        # Index yükle
        self._load_index()
    
    def _load_index(self):
        """FAISS index ve chunk'ları yükle"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
                self.index = faiss.read_index(self.index_path)
                self.chunks = np.load(self.chunks_path, allow_pickle=True)
                st.session_state.index_loaded = True
                return True
            else:
                st.warning("FAISS index bulunamadı. Lütfen önce PDF yükleyin.")
                st.session_state.index_loaded = False
                return False
        except Exception as e:
            st.error(f"Index yükleme hatası: {e}")
            st.session_state.index_loaded = False
            return False
    
    def _embed_text(self, text):
        """Metin için embedding oluştur"""
        try:
            response = requests.post(
                self.HF_API_URL,
                headers=self.HF_HEADERS,
                json={"inputs": text},
                timeout=30
            )
            response.raise_for_status()
            
            embedding = np.array(response.json(), dtype=np.float32)
            
            # Embedding boyutunu kontrol et
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            return embedding
            
        except Exception as e:
            st.error(f"Embedding oluşturma hatası: {e}")
            return None
    
    def create_index_from_pdf(self, pdf_file):
        """PDF'den FAISS index oluştur"""
        try:
            with st.spinner("📄 PDF işleniyor..."):
                # Geçici dosya oluştur
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_path = tmp_file.name
                
                # PDF'den metin çıkar
                doc = fitz.open(tmp_path)
                chunks = []
                
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text().strip()
                    if text:
                        # Sayfayı parçalara böl
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        for para in paragraphs:
                            if len(para) > 30:  # Çok kısa paragrafları atla
                                chunks.append(para)
                
                doc.close()
                os.unlink(tmp_path)  # Geçici dosyayı temizle
                
                if not chunks:
                    st.error("PDF'den metin çıkarılamadı!")
                    return False
                
                st.info(f"✅ {len(chunks)} metin parçası çıkarıldı")
            
            # Embedding oluştur
            with st.spinner("🔨 Embedding'ler oluşturuluyor..."):
                embeddings = []
                progress_bar = st.progress(0)
                
                for i, chunk in enumerate(chunks):
                    emb = self._embed_text(chunk)
                    if emb is not None:
                        embeddings.append(emb)
                    
                    # İlerleme çubuğunu güncelle
                    progress_bar.progress((i + 1) / len(chunks))
                
                if not embeddings:
                    st.error("Embedding oluşturulamadı!")
                    return False
                
                embeddings_array = np.vstack(embeddings)
            
            # FAISS index oluştur
            with st.spinner("🏗️ FAISS index oluşturuluyor..."):
                dim = embeddings_array.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(embeddings_array)
                
                # Kaydet
                faiss.write_index(index, self.index_path)
                np.save(self.chunks_path, np.array(chunks, dtype=object))
            
            # Session state'i güncelle
            self.index = index
            self.chunks = np.array(chunks, dtype=object)
            st.session_state.index_loaded = True
            
            st.success(f"✅ Index oluşturuldu ve kaydedildi: {len(chunks)} parça")
            return True
            
        except Exception as e:
            st.error(f"Index oluşturma hatası: {e}")
            return False
    
    def search(self, query, k=5):
        """Index'te benzer parçaları ara"""
        if not hasattr(self, 'index') or self.index is None:
            return []
        
        # Query embedding
        query_emb = self._embed_text(query)
        if query_emb is None:
            return []
        
        # Arama
        distances, indices = self.index.search(query_emb.reshape(1, -1), k)
        
        # Sonuçları formatla
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):  # Geçerli index kontrolü
                results.append({
                    'text': self.chunks[idx],
                    'distance': distances[0][i],
                    'similarity': 1 / (1 + distances[0][i])  # Benzerlik skoru
                })
        
        return results
    
    def ask_question(self, query, k=5):
        """Soru sor ve yanıt al"""
        # Index kontrolü
        if not st.session_state.get('index_loaded', False):
            return {
                "answer": "Lütfen önce bir PDF yükleyin ve index oluşturun.",
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
        context = "\n\n---\n\n".join([
            f"[Parça {i+1}] {result['text']}" 
            for i, result in enumerate(results)
        ])
        
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
4. "Belgede bu konu net belirtilmemiştir" gibi açık ifadeler kullan

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
                max_tokens=1024
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
        
        # PDF yükleme
        uploaded_file = st.file_uploader(
            "PDF dosyası yükleyin",
            type=['pdf'],
            help="Çevre hukuku ile ilgili PDF yükleyin"
        )
        
        if uploaded_file is not None:
            if st.button("📥 PDF'den Index Oluştur", type="primary"):
                # RAG sistemini başlat
                if 'rag_system' not in st.session_state:
                    st.session_state.rag_system = StreamlitRAGSystem()
                
                rag = st.session_state.rag_system
                
                # Index oluştur
                success = rag.create_index_from_pdf(uploaded_file)
                if success:
                    st.success("✅ Index başarıyla oluşturuldu!")
                    st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Ayarlar")
        
        k_results = st.slider(
            "Aranacak benzer doküman sayısı",
            min_value=1,
            max_value=10,
            value=3
        )
        
        st.markdown("---")
        st.markdown("### 📖 Mevcut Index")
        
        # Mevcut index durumu
        index_exists = os.path.exists("documents/index.faiss")
        chunks_exists = os.path.exists("documents/chunks.npy")
        
        if index_exists and chunks_exists:
            st.success("✅ Index yüklü")
            try:
                chunks = np.load("documents/chunks.npy", allow_pickle=True)
                st.info(f"📊 {len(chunks)} parça mevcut")
            except:
                st.info("📊 Index mevcut")
        else:
            st.warning("⚠️ Index bulunamadı")
    
    # Ana içerik alanı
    # RAG sistemini başlat
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = StreamlitRAGSystem()
    
    rag = st.session_state.rag_system
    
    # Index durumunu kontrol et
    index_loaded = st.session_state.get('index_loaded', False)
    
    if not index_loaded:
        st.warning("""
        ⚠️ **Index Yüklenmedi**
        
        Lütfen:
        1. Sidebar'dan PDF yükleyin
        2. "PDF'den Index Oluştur" butonuna tıklayın
        3. İşlemin tamamlanmasını bekleyin
        
        Veya mevcut `documents/` klasöründeki index dosyalarını kontrol edin.
        """)
    
    # Soru sorma bölümü
    st.subheader("❓ Soru Sor")
    
    query = st.text_area(
        "Çevre hukuku ile ilgili sorunuzu yazın:",
        placeholder="Örnek: Çevre kirliliği için cezai yaptırımlar nelerdir?",
        height=100,
        disabled=not index_loaded
    )
    
    if st.button("🔍 Yanıt Al", type="primary", disabled=not index_loaded) and query:
        if not index_loaded:
            st.error("Lütfen önce index oluşturun veya yükleyin!")
            return
        
        # Yanıtı al
        result = rag.ask_question(query, k=k_results)
        
        # Yanıtı göster
        st.markdown("---")
        st.subheader("🤖 Uzman Yanıtı")
        
        with st.container():
            st.markdown(result["answer"])
            
            # İstatistikler
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Güven Skoru", f"{result['confidence']:.2%}")
            with col2:
                st.metric("Kullanılan Kaynak", len(result["sources"]))
        
        # Kaynakları göster
        if result["sources"]:
            with st.expander("📚 Kullanılan Kaynaklar"):
                for i, source in enumerate(result["sources"], 1):
                    st.markdown(f"**Kaynak {i}** (Benzerlik: {source['similarity']:.2%})")
                    st.info(f"{source['text'][:400]}...")
                    st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.caption("⚡ Powered by Groq & FAISS | ⚖️ Çevre Hukuku Uzman Sistemi")

if __name__ == "__main__":
    # Environment variables kontrolü
    if not os.getenv("GROQ_API_KEY"):
        st.error("""
        ⚠️ **GROQ_API_KEY ayarlanmamış!**
        
        Lütfen aşağıdakilerden birini yapın:
        
        1. `.env` dosyası oluşturun:
        ```
        GROQ_API_KEY=your_api_key_here
        HF_TOKEN=your_huggingface_token_here
        ```
        
        2. Streamlit Cloud'da secrets kullanın:
        ```
        [secrets]
        GROQ_API_KEY = "your_api_key"
        HF_TOKEN = "your_hf_token"
        ```
        """)
    else:
        main()
