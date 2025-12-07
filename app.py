import os
import streamlit as st
import json
import time
from pathlib import Path

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="Çevre Hukuku Debug",
    page_icon="⚖️",
    layout="wide"
)

def main():
    st.title("🔍 Dosya Sistemi Kontrolü")
    st.markdown("---")
    
    # Mevcut çalışma dizini
    current_dir = os.getcwd()
    st.subheader("Mevcut Dizin")
    st.code(current_dir)
    
    # Tüm dosya ve klasörleri listele
    st.subheader("Dosya Yapısı")
    
    def list_files(startpath):
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level
            st.text(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                st.text(f'{subindent}{file}')
    
    list_files(current_dir)
    
    # Önemli dosyaları kontrol et
    st.markdown("---")
    st.subheader("📁 Önemli Dosya Kontrolleri")
    
    important_paths = [
        ("documents/", "documents/"),
        ("documents/cevre_yasasi.pdf", "PDF dosyası"),
        ("vectorstore/", "vectorstore/"),
        ("vectorstore/metadata.json", "metadata"),
        ("requirements.txt", "requirements.txt"),
        (".streamlit/", ".streamlit klasörü"),
        (".streamlit/secrets.toml", "secrets.toml"),
    ]
    
    for path, description in important_paths:
        exists = os.path.exists(path)
        status = "✅ VAR" if exists else "❌ YOK"
        
        if exists:
            if os.path.isfile(path):
                size = os.path.getsize(path)
                st.success(f"{status} - {description}: {path} ({size} bytes)")
            else:
                st.success(f"{status} - {description}: {path} (klasör)")
        else:
            st.error(f"{status} - {description}: {path}")
            
            # Eğer PDF yoksa, oluşturmak için
            if path == "documents/cevre_yasasi.pdf":
                with st.expander("PDF oluşturma seçenekleri"):
                    uploaded_file = st.file_uploader("PDF yükle", type=['pdf'])
                    if uploaded_file is not None:
                        Path("documents").mkdir(exist_ok=True)
                        with open("documents/cevre_yasasi.pdf", "wb") as f:
                            f.write(uploaded_file.getvalue())
                        st.success("PDF yüklendi! Sayfayı yenileyin.")
    
    # Environment variables kontrolü
    st.markdown("---")
    st.subheader("🔑 Environment Variables")
    
    env_vars = ["GROQ_API_KEY", "HF_TOKEN"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            st.success(f"✅ {var}: {'*' * min(8, len(value))}...")
        else:
            st.error(f"❌ {var}: AYARLANMAMIŞ")
    
    # Secrets dosyası kontrolü
    st.markdown("---")
    st.subheader("🗝️ Streamlit Secrets")
    
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r') as f:
            secrets_content = f.read()
        st.success("✅ secrets.toml bulundu")
        with st.expander("Secrets içeriği"):
            st.code(secrets_content)
    else:
        st.error("❌ secrets.toml bulunamadı")
        
        # Secrets oluşturma formu
        with st.form("create_secrets"):
            st.info("Secrets dosyası oluştur")
            groq_key = st.text_input("GROQ_API_KEY:", type="password")
            hf_token = st.text_input("HF_TOKEN:", type="password")
            
            if st.form_submit_button("Secrets Oluştur"):
                Path(".streamlit").mkdir(exist_ok=True)
                secrets_content = f'GROQ_API_KEY = "{groq_key}"\nHF_TOKEN = "{hf_token}"'
                with open(secrets_path, 'w') as f:
                    f.write(secrets_content)
                st.success("Secrets dosyası oluşturuldu! Sayfayı yenileyin.")

if __name__ == "__main__":
    main()
