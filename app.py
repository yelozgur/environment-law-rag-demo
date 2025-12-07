import os
import streamlit as st
import numpy as np
import fitz
import requests
from groq import Groq
from dotenv import load_dotenv
import faiss
import tempfile
import time
from pathlib import Path
import json

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
        
        # Dosya yolları
        self.pdf_path = "documents/cevre_yasasi.pdf"
        self.index_path = "vectorstore/index.faiss"
        self.chunks_path = "vectorstore/chunks.npy"
        self.metadata_path = "vectorstore/metadata.json"
        
        # Klasörleri oluştur
        self._ensure_directories()
        
        # Index yükle
        self._load_index()
    
    def _ensure_directories(self):
        """Gerekli klasörleri oluştur"""
        Path("documents").mkdir(exist_ok=True)
        Path("vectorstore").mkdir(exist_ok=True)
    
    def _load_index(self):
        """FAISS index ve chunk'ları yükle"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
                with st.spinner("📦 FAISS index yükleniyor..."):
                    self.index = faiss.read_index(self.index_path)
                    self.chunks = np.load(self.chunks_path, allow_pickle=True)
                    
                    # Metadata yükle (varsa)
                    if os.path.exists(self.metadata_path):
                        with open(self.metadata_path, 'r', encoding='utf-8') as f:
                            self.metadata = json.load(f)
                    else:
                        self.metadata = {"source": self.pdf_path, "chunks_count": len(self.chunks)}
                    
                st.session_state.index_loaded = True
                st.session_state.chunks_count = len(self.chunks)
                return True
            else:
                st.session_state.index_loaded = False
                return False
        except Exception as e:
            st.error(f"Index yükleme hatası: {e}")
            st.session_state.index_loaded = False
            return False
    
    def _embed_text(self, text):
        """Metin için embedding oluştur"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.HF_API_URL,
                    headers=self.HF_HEADERS,
                    json={"inputs": text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    embedding = np.array(response.json(), dtype=np.float32)
                    
                    # Embedding boyutunu kontrol et
                    if embedding.ndim == 1:
                        embedding = embedding.reshape(1, -1)
                    
                    return embedding
                elif response.status_code == 503:  # Model loading
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        time.sleep(wait_time)
                        continue
                
                response.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    st.error(f"Embedding oluşturma hatası: {e}")
                    return None
        
        return None
    
    def create_index_from_existing_pdf(self):
        """Mevcut PDF'den index oluştur"""
        if not os.path.exists(self.pdf_path):
            st.error(f"PDF dosyası bulunamadı: {self.pdf_path}")
            return False
        
        try:
            with st.spinner("📄 Mevcut PDF işleniyor..."):
                # PDF'den metin çıkar
                doc = fitz.open(self.pdf_path)
                chunks = []
                page_chunk_map = []
                
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text().strip()
                    if text:
                        # Sayfayı parçalara böl
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        for para in paragraphs:
                            if len(para) > 30:  # Çok kısa paragrafları atla
                                chunks.append(para)
                                page_chunk_map.append(page_num)
                
                doc.close()
                
                if not chunks:
                    st.error("PDF'den metin çıkarılamadı!")
                    return False
                
                st.info(f"✅ {len(chunks)} metin parçası çıkarıldı")
            
            # Embedding oluştur
            with st.spinner("🔨 Embedding'ler oluşturuluyor..."):
                embeddings = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, chunk in enumerate(chunks):
                    status_text.text(f"Parça {i+1}/{len(chunks)} işleniyor...")
                    emb = self._embed_text(chunk)
                    if emb is not None:
                        embeddings.append(emb)
                    
                    # İlerleme çubuğunu güncelle
                    progress_bar.progress((i + 1) / len(chunks))
                
                status_text.empty()
                
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
                
                # Metadata kaydet
                metadata = {
                    "source": self.pdf_path,
                    "chunks_count": len(chunks),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "page_chunk_map": page_chunk_map
                }
                
                with open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Session state'i güncelle
            self.index = index
            self.chunks = np.array(chunks, dtype=object)
            self.metadata = metadata
            st.session_state.index_loaded = True
            st.session_state.chunks_count = len(chunks)
            
            st.success(f"✅ Index oluşturuldu: {len(chunks)} parça")
            return True
            
        except Exception as e:
            st.error(f"Index oluşturma hatası: {e}")
            return False
    
    def create_index_from_new_pdf(self, pdf_file):
        """Yeni PDF'den index oluştur"""
        try:
            with st.spinner("📄 Yeni PDF işleniyor..."):
                # Geçici dosya oluştur ve kaydet
                with open(self.pdf_path, 'wb') as f:
                    f.write(pdf_file.getvalue())
                
                # PDF'den metin çıkar
                doc = fitz.open(self.pdf_path)
                chunks = []
                page_chunk_map = []
                
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text().strip()
                    if text:
                        # Sayfayı parçalara böl
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        for para in paragraphs:
                            if len(para) > 30:  # Çok kısa paragrafları atla
                                chunks.append(para)
                                page_chunk_map.append(page_num)
                
                doc.close()
                
                if not chunks:
                    st.error("PDF'den metin çıkarılamadı!")
                    return False
                
                st.info(f"✅ {len(chunks)} metin parçası çıkarıldı")
            
            # Embedding oluştur
            with st.spinner("🔨 Embedding'ler oluşturuluyor..."):
                embeddings = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, chunk in enumerate(chunks):
                    status_text.text(f"Parça {i+1}/{len(chunks)} işleniyor...")
                    emb = self._embed_text(chunk)
                    if emb is not None:
                        embeddings.append(emb)
                    
                    # İlerleme çubuğunu güncelle
                    progress_bar.progress((i + 1) / len(chunks))
                
                status_text.empty()
                
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
                
                # Metadata kaydet
                metadata = {
                    "source": self.pdf_path,
                    "filename": pdf_file.name,
                    "chunks_count": len(chunks),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "page_chunk_map": page_chunk_map
                }
                
                with open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Session state'i güncelle
            self.index = index
            self.chunks = np.array(chunks, dtype=object)
            self.metadata = metadata
            st.session_state.index_loaded = True
            st.session_state.chunks_count = len(chunks)
            
            st.success(f"✅ Index oluşturuldu: {len(chunks)} parça")
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
                # Sayfa numarasını bul (varsa)
                page_num = self.metadata.get("page_chunk_map", [])[idx] if "page_chunk_map" in self.metadata else None
                
                results.append({
                    'text': self.chunks[idx],
                    'distance': float(distances[0][i]),
                    'similarity': 1 / (1 + distances[0][i]),  # Benzerlik skoru
                    'page': page_num,
                    'chunk_id': int(idx)
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
        context_parts = []
        for i, result in enumerate(results):
            page_info = f" [Sayfa {result['page']}]" if result['page'] else ""
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
            
            if st.button("🔄 Mevcut PDF'den Index Oluştur", type="primary"):
                if 'rag_system' not in st.session_state:
                    st.session_state.rag_system = StreamlitRAGSystem()
                
                rag = st.session_state.rag_system
                success = rag.create_index_from_existing_pdf()
                if success:
                    st.success("✅ Index başarıyla oluşturuldu!")
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
            if st.button("📥 Yeni PDF ile Index Oluştur", type="secondary"):
                if 'rag_system' not in st.session_state:
                    st.session_state.rag_system = StreamlitRAGSystem()
                
                rag = st.session_state.rag_system
                success = rag.create_index_from_new_pdf(uploaded_file)
                if success:
                    st.success("✅ Yeni PDF ile index oluşturuldu!")
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
        
        # Index durumu
        index_exists = os.path.exists("vectorstore/index.faiss")
        chunks_exists = os.path.exists("vectorstore/chunks.npy")
        
        if index_exists and chunks_exists:
            st.success("✅ Index yüklü")
            try:
                chunks = np.load("vectorstore/chunks.npy", allow_pickle=True)
                st.info(f"📊 {len(chunks)} parça")
                
                # Metadata göster
                if os.path.exists("vectorstore/metadata.json"):
                    with open("vectorstore/metadata.json", 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    st.caption(f"Kaynak: {os.path.basename(metadata.get('source', 'Unknown'))}")
                    st.caption(f"Oluşturulma: {metadata.get('created_at', 'Unknown')}")
            except:
                st.info("📊 Vector store mevcut")
        else:
            st.warning("⚠️ Index bulunamadı")
        
        # Temizleme butonu
        st.markdown("---")
        if st.button("🗑️ Vector Store'u Temizle", type="secondary"):
            try:
                for file in ["vectorstore/index.faiss", "vectorstore/chunks.npy", "vectorstore/metadata.json"]:
                    if os.path.exists(file):
                        os.remove(file)
                st.success("✅ Vector store temizlendi!")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"Temizleme hatası: {e}")
    
    # Ana içerik alanı
    # RAG sistemini başlat
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = StreamlitRAGSystem()
    
    rag = st.session_state.rag_system
    
    # Index durumunu kontrol et
    index_loaded = st.session_state.get('index_loaded', False)
    
    if not index_loaded:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.warning("""
            ### ⚠️ Vector Store Yüklenemedi
            
            **Mevcut Durum:**
            - PDF: `documents/cevre_yasasi.pdf` - {'✅ Mevcut' if pdf_exists else '❌ Eksik'}
            - Index: `vectorstore/index.faiss` - {'✅ Mevcut' if index_exists else '❌ Eksik'}
            - Chunks: `vectorstore/chunks.npy` - {'✅ Mevcut' if chunks_exists else '❌ Eksik'}
            
            **Ne yapabilirsiniz:**
            1. **Mevcut PDF'den index oluştur** → Sidebar'daki butonu kullanın
            2. **Yeni PDF yükle** → Sidebar'dan yeni PDF yükleyin
            3. **Manuel kontrol** → Dosyaların doğru yerde olduğundan emin olun
            """)
        
        with col2:
            st.info("""
            **📁 Dosya Yapısı:**
            ```
            main/
            ├── documents/
            │   └── cevre_yasasi.pdf
            ├── vectorstore/
            │   ├── index.faiss
            │   ├── chunks.npy
            │   └── metadata.json
            ├── app.py
            └── requirements.txt
            ```
            """)
    
    # Soru sorma bölümü
    st.subheader("❓ Soru Sor")
    
    query = st.text_area(
        "Çevre hukuku ile ilgili sorunuzu yazın:",
        placeholder="Örnek: Çevre kirliliği için cezai yaptırımlar nelerdir? Atık yönetimi yükümlülükleri nelerdir? Çevre izinleri nasıl alınır?",
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
                    avg_page = np.mean([s.get('page', 0) for s in result['sources'] if s.get('page')])
                    if avg_page > 0:
                        st.metric("Ort. Sayfa No", f"{avg_page:.0f}")
            
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
        st.caption("🔍 FAISS Vector Search")
    with col3:
        st.caption("⚖️ Çevre Hukuku Uzmanı")

if __name__ == "__main__":
    # Environment variables kontrolü
    groq_key = os.getenv("GROQ_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    
    if not groq_key:
        st.error("""
        ### ⚠️ GROQ_API_KEY ayarlanmamış!
        
        **Çözüm yolları:**
        
        1. **Streamlit Cloud Secrets:**
           ```toml
           # .streamlit/secrets.toml
           GROQ_API_KEY = "sk-..."
           HF_TOKEN = "hf_..."
           ```
        
        2. **Local .env dosyası:**
           ```bash
           # .env dosyası oluşturun
           GROQ_API_KEY=sk-...
           HF_TOKEN=hf_...
           ```
        
        3. **Manuel giriş (geliştirme için):**
        """)
        
        # Geliştirme için manuel giriş
        with st.form("api_keys_form"):
            groq_input = st.text_input("GROQ API Key:", type="password")
            hf_input = st.text_input("HuggingFace Token:", type="password")
            
            if st.form_submit_button("API Key'leri Kaydet"):
                os.environ["GROQ_API_KEY"] = groq_input
                os.environ["HF_TOKEN"] = hf_input
                st.success("API Key'ler kaydedildi! Sayfayı yenileyin.")
                st.rerun()
    else:
        main()
