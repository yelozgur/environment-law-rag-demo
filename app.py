import streamlit as st
import fitz  # PyMuPDF
from rag import RAG


# ----- Page config -----
st.set_page_config(page_title="Çevre Hukuku Danışma Asistanı", layout="wide")

st.title("⚖️ Çevre Hukuku Danışma Hattı – AI Destekli Demo")

st.write("""
Bu demo, Kıbrıs çevre mevzuatına ilişkin sorular için 
LLM + RAG (Belge Tabanlı Arama) mimarisi kullanır.
""")


# ----- Document loader -----
def load_pdf(path):
    """PDF'i parçalara bölüp RAG'e iletilecek belge formatına dönüştürür."""
    doc = fitz.open(path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > 20:
            pages.append({"id": f"p{i+1}", "text": text})

    return pages


# ----- RAG Loader -----
@st.cache_resource
def load_rag():
    api_key = st.secrets["GEMINI_API_KEY"]

    documents = load_pdf("documents/cevre_yasasi.pdf")

    return RAG(api_key=api_key, documents=documents)


rag = load_rag()


# ----- User Query -----
query = st.text_input("Bir soru yazın (örn: ÇED gerektiren projeler nelerdir?)")

if st.button("Sorgula"):
    if not query.strip():
        st.error("Lütfen bir soru yazın.")
    else:
        with st.spinner("Yanıt hazırlanıyor..."):
            answer, ctx = rag.ask(query)

            st.subheader("📌 Yanıt")
            st.write(answer)

            with st.expander("📄 Kaynaklar"):
                for c in ctx:
                    st.write(f"**{c['id']}**: {c['text'][:400]}...")
