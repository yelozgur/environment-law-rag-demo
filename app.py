import streamlit as st
from rag import RAG

st.set_page_config(page_title="Çevre Hukuku Danışma Asistanı", layout="wide")

st.title("⚖️ Çevre Hukuku Danışma Hattı – AI Destekli Demo")

st.write("""
Bu demo, Kıbrıs çevre mevzuatına ilişkin sorular için 
LLM + RAG (Belge Tabanlı Arama) mimarisi kullanır.
""")

@st.cache_resource
def load_rag():
    return RAG("documents/cevre_yasasi.pdf")

rag = load_rag()

query = st.text_input("Bir soru yazın (örn: ÇED gerektiren projeler nelerdir?)")

if st.button("Sorgula"):
    if not query.strip():
        st.error("Lütfen bir soru yazın.")
    else:
        with st.spinner("Yanıt hazırlanıyor..."):
            answer = rag.ask_lawyer(query)
            st.subheader("📌 Yanıt")
            st.write(answer)
