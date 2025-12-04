
import streamlit as st
import os
# Inject Custom CSS
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white !important;
}
.main {
    background: transparent !important;
}
.block-container {
    background: rgba(255, 255, 255, 0.1);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
h1, h2, h3, h4, h5, h6 {
    color: #f5f7fa !important;
    text-shadow: 0px 1px 3px rgba(0,0,0,0.4);
}
.stButton>button {
    background-color: #ff7e5f;
    color: white !important;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #feb47b;
    transform: scale(1.05);
}
.uploadedFile {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
st.title("🌈 AI Article → Image Generator")
st.markdown("### Transform Entire Articles into Beautiful High-Resolution Images 📸✨")
mode = st.radio("Choose Mode", ["Single Article", "Batch Mode"])
from main import extract_text_from_uploaded, summarize_article, extract_characters, build_prompts_from_summary, generate_images_for_prompts
if mode == "Single Article":
    file = st.file_uploader("📄 Upload Article File", type=["txt", "pdf", "docx"])
    if file:
        with st.spinner("Reading file..."):
            text = extract_text_from_uploaded(file, file.name)
        st.subheader("📘 Article Preview")
        st.write(text[:1000] + ("..." if len(text) > 1000 else ""))
        if st.button("Generate Images"):
            with st.spinner("Summarizing article..."):
                summary = summarize_article(text)
                characters = extract_characters(text)
                prompts = build_prompts_from_summary(summary, characters)
            st.subheader("🧠 Generated Prompts")
            for i, p in enumerate(prompts):
                st.markdown(f"**Frame {i+1}:** {p}")
            with st.spinner("Creating images..."):
                images = generate_images_for_prompts(prompts, article_name=file.name.split('.')[0])
            st.success("🎉 Image Generation Complete!")
            for img_path in images:
                st.image(img_path, width=450)
else:
    files = st.file_uploader("📄 Upload Multiple Files", type=["txt","pdf","docx"], accept_multiple_files=True)
    if files and st.button("Process All"):
        for file in files:
            st.write(f"➡ Processing **{file.name}**...")
            text = extract_text_from_uploaded(file, file.name)
            summary = summarize_article(text)
            characters = extract_characters(text)
            prompts = build_prompts_from_summary(summary, characters)
            images = generate_images_for_prompts(prompts, article_name=file.name.split('.')[0])
        st.success("🎉 Batch Processing Complete! All images saved to Google Drive.")
